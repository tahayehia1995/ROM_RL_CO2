"""
Checkpoint Catalog
==================
Read-only index over the ``saved_models/`` tree, used by the grid search
to discover compatible warm-start weights for ``use_pretrained=True``
runs.

The catalog scans every ``e2co_encoder_*.h5`` file, parses its filename
(which encodes the embedder family, latent/channel dims, residual
blocks, normalization, transition type, and transition encoder hidden
dims), groups it with its sibling ``decoder``/``transition`` files, and
exposes two strict same-family lookups:

- ``find_compatible_encoder_decoder(target)`` -> (encoder_path, decoder_path)
- ``find_compatible_transition(target)``     -> transition_path

When several candidates match a target, the most recently modified file
wins (mtime tiebreak).

Filename schema (matches ``grid_search_training.create_run_id`` plus
``create_run_model_filename``):

    e2co_<component>_grid_run<idx>_bs<B>_ld<L>_ns<N>_ch<C>_sch<sched>
        _rb<R>_norm<ba|gd>_ehd<H1-H2-...>[_<flags>]_<family>[_trn<TYPE>].h5

where:
    <component> in {encoder, decoder, transition}
    <flags>     subset of {fft, mask, dlw<...>, adv}
    <family>    one of {gnn, mm, fno, mem-<preset-with-hyphens>}
                (mem family also has a separate `memT` token)
    <TYPE>      transition type, upper-cased; omitted when 'linear'
                (matches the existing case-sensitive check in
                ``create_run_id``)

The catalog is intentionally pure-Python on the scan path -- torch is
only imported for optional payload-sentinel verification when filename
parsing leaves any ambiguity.
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Family tokens recognised in filenames.  Order matters: longer tokens
# (mem-...) must be checked before shorter ones (mm, fno) to avoid
# ambiguous prefix matches.
# ---------------------------------------------------------------------------
_FAMILY_TOKENS = ('gnn', 'mm', 'fno')   # 'mem-...' handled separately


# ---------------------------------------------------------------------------
# Per-checkpoint metadata
# ---------------------------------------------------------------------------
@dataclass
class CheckpointMeta:
    """Parsed metadata for a single (encoder, decoder, transition) trio."""
    family: str                       # 'gnn' | 'mm' | 'fno' | 'mem'
    mem_preset: Optional[str] = None  # e.g. 'cnn-sg--fno-pres'
    latent_dim: int = 0
    n_channels: int = 0
    n_steps: int = 0
    residual_blocks: Optional[int] = None
    norm_type: Optional[str] = None       # 'batchnorm' | 'gdn'
    encoder_hidden_dims: Optional[Tuple[int, ...]] = None
    transition_type: str = 'linear'       # lower-case canonical form
    encoder_path: str = ''
    decoder_path: Optional[str] = None
    transition_path: Optional[str] = None
    mtime: float = 0.0
    raw_run_id: str = ''
    flags: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Target signature: what the caller is about to train
# ---------------------------------------------------------------------------
@dataclass
class TargetSignature:
    """Identifies what a run is about to train.  Built from the grid
    search hyperparameter dict via :func:`build_target_signature`.
    """
    family: str                            # 'gnn' | 'mm' | 'fno' | 'mem'
    mem_preset: Optional[str] = None       # canonical preset name (with hyphens)
    latent_dim: int = 0
    n_channels: int = 0
    n_steps: int = 0
    residual_blocks: Optional[int] = None
    norm_type: Optional[str] = None
    encoder_hidden_dims: Optional[Tuple[int, ...]] = None
    transition_type: str = 'linear'        # lower-case canonical form


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
_TOKEN_PATTERNS = {
    'run':   re.compile(r'^run(\d+)$'),
    'bs':    re.compile(r'^bs(\d+)$'),
    'ld':    re.compile(r'^ld(\d+)$'),
    'ns':    re.compile(r'^ns(\d+)$'),
    'ch':    re.compile(r'^ch(\d+)$'),
    'sch':   re.compile(r'^sch([a-z0-9]+)$'),
    'rb':    re.compile(r'^rb(\d+)$'),
    'norm':  re.compile(r'^norm([a-z]+)$'),
    'ehd':   re.compile(r'^ehd([\d\-]+)$'),
    'mem':   re.compile(r'^mem-([a-zA-Z0-9\-]+)$'),
    'trn':   re.compile(r'^trn([A-Za-z0-9_]+)$'),
}


def _norm_token_to_full(t: str) -> str:
    if t == 'ba':
        return 'batchnorm'
    if t == 'gd':
        return 'gdn'
    return t


def parse_encoder_filename(filename: str) -> Optional[CheckpointMeta]:
    """Parse a single ``e2co_encoder_grid_*.h5`` basename into metadata.

    Returns ``None`` if the file does not match the expected pattern
    (e.g. an unrelated ``.h5`` that happens to live in saved_models).
    """
    base = os.path.basename(filename)
    if not base.startswith('e2co_encoder_grid_') or not base.endswith('.h5'):
        return None

    run_id = base[len('e2co_encoder_grid_'):-len('.h5')]
    tokens = run_id.split('_')

    meta = CheckpointMeta(family='', encoder_path=filename, raw_run_id=run_id)
    flags: List[str] = []
    family_seen = False

    for tok in tokens:
        m = _TOKEN_PATTERNS['run'].match(tok)
        if m:
            continue
        m = _TOKEN_PATTERNS['bs'].match(tok)
        if m:
            continue  # batch size not relevant to compatibility
        m = _TOKEN_PATTERNS['ld'].match(tok)
        if m:
            meta.latent_dim = int(m.group(1)); continue
        m = _TOKEN_PATTERNS['ns'].match(tok)
        if m:
            meta.n_steps = int(m.group(1)); continue
        m = _TOKEN_PATTERNS['ch'].match(tok)
        if m:
            meta.n_channels = int(m.group(1)); continue
        m = _TOKEN_PATTERNS['sch'].match(tok)
        if m:
            continue  # scheduler not relevant
        m = _TOKEN_PATTERNS['rb'].match(tok)
        if m:
            meta.residual_blocks = int(m.group(1)); continue
        m = _TOKEN_PATTERNS['norm'].match(tok)
        if m:
            meta.norm_type = _norm_token_to_full(m.group(1)); continue
        m = _TOKEN_PATTERNS['ehd'].match(tok)
        if m:
            meta.encoder_hidden_dims = tuple(int(x) for x in m.group(1).split('-'))
            continue
        m = _TOKEN_PATTERNS['mem'].match(tok)
        if m:
            meta.family = 'mem'
            meta.mem_preset = m.group(1)
            family_seen = True
            continue
        if tok == 'memT':
            # Marker added alongside mem-<preset>; carries no extra info.
            continue
        m = _TOKEN_PATTERNS['trn'].match(tok)
        if m:
            meta.transition_type = m.group(1).lower()
            continue
        if tok in _FAMILY_TOKENS:
            # First family token wins; if MEM was already seen we ignore
            # any subsequent gnn/mm/fno tokens (defensive).
            if not family_seen:
                meta.family = tok
                family_seen = True
            continue
        # Anything else is a flag (fft, mask, dlw..., adv, pt, ...).
        flags.append(tok)

    meta.flags = tuple(flags)

    if not family_seen:
        # Plain MSE2C / unfamiliar layout -- skip from catalog.
        return None

    try:
        meta.mtime = os.path.getmtime(filename)
    except OSError:
        meta.mtime = 0.0
    return meta


def _sibling_path(encoder_path: str, component: str) -> str:
    """Return the sibling decoder/transition path for an encoder file."""
    base = os.path.basename(encoder_path)
    new_base = base.replace('e2co_encoder_grid_', f'e2co_{component}_grid_', 1)
    return os.path.join(os.path.dirname(encoder_path), new_base)


# ---------------------------------------------------------------------------
# CheckpointCatalog
# ---------------------------------------------------------------------------
class CheckpointCatalog:
    """In-memory index over saved_models/ for warm-start discovery."""

    def __init__(self, search_dirs):
        if isinstance(search_dirs, (str, bytes)):
            search_dirs = [search_dirs]
        self.search_dirs = [str(d) for d in search_dirs]
        self._entries: List[CheckpointMeta] = []
        self._scan()

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------
    def _scan(self) -> None:
        seen = set()
        for d in self.search_dirs:
            if not os.path.isdir(d):
                continue
            for path in glob.glob(os.path.join(d, 'e2co_encoder_grid_*.h5')):
                resolved = os.path.normpath(os.path.abspath(path))
                if resolved in seen:
                    continue
                seen.add(resolved)
                meta = parse_encoder_filename(resolved)
                if meta is None:
                    continue
                # Attach sibling paths if they exist on disk.  Missing
                # siblings simply mean that component is not warm-startable
                # from this run.
                dec = _sibling_path(resolved, 'decoder')
                trn = _sibling_path(resolved, 'transition')
                meta.decoder_path = dec if os.path.isfile(dec) else None
                meta.transition_path = trn if os.path.isfile(trn) else None
                self._entries.append(meta)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def all_entries(self) -> List[CheckpointMeta]:
        return list(self._entries)

    def find_compatible_encoder_decoder(
        self, target: TargetSignature
    ) -> Optional[Tuple[str, str]]:
        """Return ``(encoder_path, decoder_path)`` for the most recent
        same-family checkpoint whose encoder/decoder shape signature
        matches ``target``.

        Strict same-family match: ``family``, ``latent_dim``,
        ``n_channels``, ``residual_blocks``, ``norm_type`` and (for MEM
        only) ``mem_preset``.  Both encoder and decoder files must exist
        on disk.
        """
        candidates = [
            m for m in self._entries
            if self._enc_dec_compatible(m, target)
            and m.decoder_path is not None
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda m: m.mtime)
        return (best.encoder_path, best.decoder_path)

    def find_compatible_transition(
        self, target: TargetSignature
    ) -> Optional[str]:
        """Return the path of the most recent same-family transition
        checkpoint that matches the target's transition class and
        latent/encoder-hidden geometry.
        """
        candidates = [
            m for m in self._entries
            if self._trans_compatible(m, target)
            and m.transition_path is not None
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda m: m.mtime)
        return best.transition_path

    # ------------------------------------------------------------------
    # Compatibility predicates
    # ------------------------------------------------------------------
    @staticmethod
    def _enc_dec_compatible(m: CheckpointMeta, t: TargetSignature) -> bool:
        if m.family != t.family:
            return False
        if m.latent_dim != t.latent_dim:
            return False
        if m.n_channels != t.n_channels:
            return False
        if t.residual_blocks is not None and m.residual_blocks != t.residual_blocks:
            return False
        if t.norm_type is not None and m.norm_type != t.norm_type:
            return False
        if t.family == 'mem':
            if (m.mem_preset or '') != (t.mem_preset or ''):
                return False
        return True

    @staticmethod
    def _trans_compatible(m: CheckpointMeta, t: TargetSignature) -> bool:
        if m.family != t.family:
            return False
        if m.transition_type.lower() != t.transition_type.lower():
            return False
        if m.latent_dim != t.latent_dim:
            return False
        # Encoder hidden dims drive the transition's input MLP shape.
        if (t.encoder_hidden_dims is not None
                and m.encoder_hidden_dims != t.encoder_hidden_dims):
            return False
        if t.family == 'mem':
            if (m.mem_preset or '') != (t.mem_preset or ''):
                return False
        return True


# ---------------------------------------------------------------------------
# Hyperparam dict -> TargetSignature
# ---------------------------------------------------------------------------
def build_target_signature(hyperparams: Dict) -> Optional[TargetSignature]:
    """Translate a grid-search hyperparam dict into a :class:`TargetSignature`.

    Returns ``None`` if no supported embedder family is enabled (e.g.
    plain MSE2C runs).
    """
    if hyperparams.get('enable_multi_embedding_multimodal', False):
        family = 'mem'
        preset_raw = hyperparams.get('mem_branches_preset', 'unknown')
        # Filename uses underscores -> hyphens (see grid_search_training).
        mem_preset = preset_raw.replace('_', '-')
    elif hyperparams.get('enable_gnn', False):
        family, mem_preset = 'gnn', None
    elif hyperparams.get('enable_fno', False):
        family, mem_preset = 'fno', None
    elif hyperparams.get('enable_multimodal', False):
        family, mem_preset = 'mm', None
    else:
        return None

    norm_type = hyperparams.get('norm_type', 'batchnorm')
    ehd_raw = hyperparams.get('encoder_hidden_dims')
    ehd: Optional[Tuple[int, ...]] = (
        tuple(int(x) for x in ehd_raw) if ehd_raw is not None else None
    )

    return TargetSignature(
        family=family,
        mem_preset=mem_preset,
        latent_dim=int(hyperparams.get('latent_dim', 0)),
        n_channels=int(hyperparams.get('n_channels', 0)),
        n_steps=int(hyperparams.get('n_steps', 0)),
        residual_blocks=hyperparams.get('residual_blocks'),
        norm_type=norm_type,
        encoder_hidden_dims=ehd,
        transition_type=str(hyperparams.get('transition_type', 'linear')).lower(),
    )
