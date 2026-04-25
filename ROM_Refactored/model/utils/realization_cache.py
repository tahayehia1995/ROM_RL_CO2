"""
Realization-Aware Static Latent Cache
======================================
Multi-entry, realization-keyed cache for the per-sample static latent vector
``z_static`` shared by all multimodal-style E2C models.

Replaces the legacy single-entry tensor-identity cache (one entry per model
that gets clobbered on any geology change) with an LRU map keyed by a
realization identifier.  This lets a single inference run cycle through
many realizations without thrashing.

Two key derivers are exposed so the cache works equally well for:

1. Models that already know the case-to-realization mapping (the GNN, via
   ``GraphManager.case_to_realization``) -- use ``keys_from_case_indices``.
2. Models with no upstream mapping (Multimodal, MultiEmbedding) where the
   inference APIs only see the spatial tensor -- use
   ``keys_from_static_tensor`` which fingerprints each sample's static
   channels via a fast deterministic hash.

Both paths return a list of integer keys (one per batch sample), so the
caller logic is identical:

    keys = cache.keys_from_<...>(...)
    z_static, miss_idx, miss_keys = cache.lookup(keys, dim=latent_dim, device=...)
    if miss_idx.numel() > 0:
        # encode only cache misses
        z_miss = encoder(x_static[miss_idx])
        z_static[miss_idx] = z_miss
        cache.store(miss_keys, z_miss)
    return z_static
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tensor fingerprinting
# ---------------------------------------------------------------------------
def _fingerprint_static_sample(sample: torch.Tensor) -> int:
    """Return a deterministic 63-bit integer fingerprint for a single static
    sample tensor.

    Uses ``hashlib.blake2b`` over the raw float32 bytes of the tensor.
    Blake2b is a C-implemented cryptographic hash and is roughly two
    orders of magnitude faster than a pure-Python byte loop, which is the
    difference between sub-millisecond and tens-of-milliseconds per
    call for typical reservoir static tensors (~100 KB).

    Two samples with bit-identical static content map to the same key;
    any change (even to a single voxel) produces a new key with
    overwhelming probability.  This path is only used as a fallback when
    callers cannot supply ``case_indices``; performance-critical sites
    should prefer the case-index path.
    """
    if sample.dtype != torch.float32:
        sample = sample.float()
    arr = sample.detach().cpu().contiguous().numpy().tobytes()
    digest = hashlib.blake2b(arr, digest_size=8).digest()
    return int.from_bytes(digest, 'big', signed=False) & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# RealizationCache
# ---------------------------------------------------------------------------
class RealizationCache:
    """LRU multi-entry cache mapping realization id -> z_static tensor.

    Each cache entry stores a single 1-D ``(latent_dim,)`` tensor (the
    encoded static latent for one realization), detached so no gradient
    flows back through cached entries.

    Thread safety: not required (models hold one cache per instance and
    are not shared across processes).
    """

    def __init__(self, maxlen: int = 64):
        self.maxlen = int(maxlen)
        self._store: "OrderedDict[int, torch.Tensor]" = OrderedDict()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop all cached entries.  Called at the start of every
        training-mode forward so weight updates are not shadowed by stale
        latents.
        """
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------
    @staticmethod
    def keys_from_case_indices(
        case_indices: Optional[torch.Tensor],
        case_to_realization: Optional[Sequence[int]],
        batch_size: int,
    ) -> List[int]:
        """Map each batch sample to a realization id using a precomputed
        ``case_to_realization`` array (provided by ``GraphManager``).

        Falls back to "all realization 0" when either ``case_indices`` or
        the mapping is unavailable.
        """
        if case_indices is None or case_to_realization is None:
            return [0] * batch_size
        out: List[int] = []
        for ci in case_indices:
            ci_int = int(ci)
            if ci_int < len(case_to_realization):
                out.append(int(case_to_realization[ci_int]))
            else:
                out.append(0)
        return out

    @staticmethod
    def keys_from_static_tensor(x_static: torch.Tensor) -> List[int]:
        """Fingerprint each sample's static spatial tensor.

        ``x_static`` shape: ``(B, C_static, Nx, Ny, Nz)``.

        Returns one integer key per batch sample.  Identical static
        content across two samples yields the same key (so they share
        cache entries even if no upstream realization mapping exists).
        """
        if x_static.dim() < 2:
            raise ValueError(
                f"keys_from_static_tensor expected at least 2-D tensor "
                f"(B, ...); got shape {tuple(x_static.shape)}"
            )
        return [_fingerprint_static_sample(x_static[i]) for i in range(x_static.shape[0])]

    @staticmethod
    def derive_keys(
        x_static: Optional[torch.Tensor],
        case_indices: Optional[torch.Tensor] = None,
        case_to_realization: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> List[int]:
        """Pick the most efficient cache key derivation available.

        Preference order:
          1. ``case_indices`` + ``case_to_realization`` mapping
             -> realization id (cases sharing geology share cache entry).
             This matches the GNN's behaviour exactly.
          2. ``case_indices`` only (no mapping)
             -> use the case index itself as the cache key.  Two cases
             with identical geology will get separate entries, but this
             is still correct and avoids the slow tensor-fingerprint.
          3. ``x_static`` only (no case info)
             -> per-sample blake2b fingerprint.  Slowest path; used as
             a last-resort safety net for callers that cannot supply
             case_indices.

        The chosen path is transparent to the caller -- it always
        receives a list of integer keys, one per sample.
        """
        if case_indices is not None and case_to_realization is not None:
            B = case_indices.shape[0] if hasattr(case_indices, 'shape') else len(case_indices)
            return RealizationCache.keys_from_case_indices(
                case_indices, case_to_realization, B
            )
        if case_indices is not None:
            return [int(ci) for ci in case_indices]
        if x_static is not None:
            return RealizationCache.keys_from_static_tensor(x_static)
        if batch_size is None:
            raise ValueError(
                "derive_keys: at least one of case_indices, x_static, or "
                "batch_size must be provided."
            )
        # Defensive fallback: single-realization assumption.
        return [0] * batch_size

    # ------------------------------------------------------------------
    # Lookup / store
    # ------------------------------------------------------------------
    def lookup(
        self,
        keys: Sequence[int],
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Build a per-sample stack of cached latents and report which
        samples missed.

        Args:
            keys: realization keys, one per batch sample.
            latent_dim: dimensionality of z_static.
            device: device to allocate the output tensor on.
            dtype: dtype for the output tensor.

        Returns:
            z_static: (B, latent_dim) tensor pre-filled with cached hits;
                      cache-miss rows are zero (caller must overwrite).
            miss_idx: (M,) int64 tensor of batch positions that missed.
            miss_keys: list of length M with the corresponding keys; in
                       the same order as ``miss_idx`` so callers can pair
                       newly-encoded latents with their keys via
                       ``store(miss_keys, z_new)``.
        """
        B = len(keys)
        z_static = torch.zeros(B, latent_dim, device=device, dtype=dtype)
        miss_positions: List[int] = []
        miss_keys: List[int] = []
        for i, k in enumerate(keys):
            if k in self._store:
                cached = self._store[k]
                z_static[i] = cached.to(device=device, dtype=dtype)
                self._store.move_to_end(k)
            else:
                miss_positions.append(i)
                miss_keys.append(k)
        miss_idx = torch.tensor(miss_positions, dtype=torch.long, device=device)
        return z_static, miss_idx, miss_keys

    def store(self, miss_keys: Sequence[int], z_new: torch.Tensor) -> None:
        """Persist newly-encoded latents.

        ``z_new`` shape must be ``(M, latent_dim)`` where ``M ==
        len(miss_keys)``.  Each row is detached and stored under its key.
        """
        if z_new.shape[0] != len(miss_keys):
            raise ValueError(
                f"store(): got {z_new.shape[0]} latents but {len(miss_keys)} keys"
            )
        for i, k in enumerate(miss_keys):
            # If two samples in the same miss batch share a key (possible
            # when fingerprints collide or when callers passed duplicates),
            # the later write wins -- they should be identical anyway.
            self._store[k] = z_new[i].detach()
            self._store.move_to_end(k)
            if len(self._store) > self.maxlen:
                self._store.popitem(last=False)


# ---------------------------------------------------------------------------
# Realization fingerprinting from raw arrays
# ---------------------------------------------------------------------------
def case_to_realization_from_static_tensor(
    x_static: torch.Tensor,
) -> List[int]:
    """Build a case-index -> realization-id list by fingerprinting each
    sample's static-channel slice.

    Two cases that produce the same blake2b fingerprint over their
    static channels are treated as the same realization and will share
    a cache entry in :class:`RealizationCache` (matching the GNN's
    ``case_to_realization`` semantics).

    Args:
        x_static: (num_cases, C_static, ...) tensor of static channels
                  for every case in the dataset (typically extracted as
                  ``state[:, 0, static_channels, ...]`` from the first
                  timestep).

    Returns:
        list of length ``num_cases`` whose ``i``-th entry is the
        realization id assigned to case ``i``.  Realization ids are
        assigned in order of first appearance.
    """
    if x_static.dim() < 2:
        raise ValueError(
            f"case_to_realization_from_static_tensor expected a tensor "
            f"of shape (num_cases, ...); got {tuple(x_static.shape)}"
        )
    fp_to_real = {}
    out: List[int] = []
    for i in range(x_static.shape[0]):
        fp = _fingerprint_static_sample(x_static[i])
        if fp not in fp_to_real:
            fp_to_real[fp] = len(fp_to_real)
        out.append(fp_to_real[fp])
    return out


def identify_realizations_from_arrays(
    masks: np.ndarray,
    permi: np.ndarray,
    num_cases: int,
) -> Tuple[List[int], np.ndarray]:
    """Group cases by unique ``(active_mask, PERMI)`` combinations.

    This is a thin shared helper extracted from the GNN's
    ``GraphManager._identify_realizations`` so other code paths (e.g. the
    ``RealizationCache``'s tensor-hash construction or future analytics)
    can reuse the exact same realization definition without depending on
    the GNN module.

    Args:
        masks: (num_cases, ...) bool array; first axis is the case axis.
        permi: (num_cases, ...) float array; first axis is the case axis.
        num_cases: number of cases.

    Returns:
        representative_cases: list of one canonical case index per unique
            realization.
        case_to_real: (num_cases,) int64 array mapping each case index to
            the realization index it belongs to.
    """
    case_to_real = np.zeros(num_cases, dtype=np.int64)
    representative_cases = [0]
    case_to_real[0] = 0

    for c in range(1, num_cases):
        found = False
        for r_idx, rep in enumerate(representative_cases):
            if (np.array_equal(masks[c], masks[rep]) and
                    np.allclose(permi[c], permi[rep], rtol=1e-5)):
                case_to_real[c] = r_idx
                found = True
                break
        if not found:
            case_to_real[c] = len(representative_cases)
            representative_cases.append(c)

    return representative_cases, case_to_real
