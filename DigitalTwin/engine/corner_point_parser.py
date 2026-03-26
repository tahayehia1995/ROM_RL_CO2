"""
CMG Corner-Point Grid Parser
==============================
Parses the CORNERS block from a CMG GEM/STARS ``.dat`` simulation data file
and produces a structured corner-coordinate array suitable for building a
PyVista hexahedral mesh.

CMG CORNERS format for ``GRID CORNER NI NJ NK``
------------------------------------------------
Three consecutive coordinate blocks (X, Y, Z), each containing
``2*NI * 2*NJ * 2*NK`` scalar values stored I-fastest, J-next, K-slowest.

For cell ``(i, j, k)``, the 8 corner coordinates live at indices
``[2*k:2*k+2, 2*j:2*j+2, 2*i:2*i+2]`` within each coordinate block.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REPEAT_RE = re.compile(r"(\d+)\*([+-]?[\d.eE+\-]+)")


def _expand_cmg_tokens(line: str) -> list:
    """Expand CMG repeat notation in a single line.

    ``3*1.5`` becomes ``[1.5, 1.5, 1.5]``.
    Plain numbers pass through unchanged.
    Comment lines (starting with ``**``) are skipped.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("**"):
        return []

    values = []
    for token in stripped.split():
        if token.startswith("**"):
            break
        m = _REPEAT_RE.fullmatch(token)
        if m:
            count = int(m.group(1))
            val = float(m.group(2))
            values.extend([val] * count)
        else:
            try:
                values.append(float(token))
            except ValueError:
                break
    return values


def parse_corners_from_file(
    dat_path: str,
    ni: int = 34,
    nj: int = 16,
    nk: int = 25,
) -> np.ndarray:
    """Parse the CORNERS block and return an ``(NI, NJ, NK, 8, 3)`` array.

    Parameters
    ----------
    dat_path : str
        Path to the CMG ``.dat`` file containing ``GRID CORNER`` and ``CORNERS``.
    ni, nj, nk : int
        Grid dimensions (must match the ``GRID CORNER`` declaration).

    Returns
    -------
    corners : np.ndarray, shape ``(ni, nj, nk, 8, 3)``
        For each cell ``(i, j, k)``, the ``(x, y, z)`` of its 8 hexahedral
        corners ordered as::

            0: left-front-bottom   (2i,   2j,   2k  )
            1: right-front-bottom  (2i+1, 2j,   2k  )
            2: left-back-bottom    (2i,   2j+1, 2k  )
            3: right-back-bottom   (2i+1, 2j+1, 2k  )
            4: left-front-top      (2i,   2j,   2k+1)
            5: right-front-top     (2i+1, 2j,   2k+1)
            6: left-back-top       (2i,   2j+1, 2k+1)
            7: right-back-top      (2i+1, 2j+1, 2k+1)
    """
    vals_per_block = 2 * ni * 2 * nj * 2 * nk
    total_expected = 3 * vals_per_block

    all_values: list = []
    in_corners = False

    with open(dat_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not in_corners:
                if line.startswith("CORNERS"):
                    in_corners = True
                continue

            upper = line.lstrip("*").strip().split()[0] if line.strip() else ""
            if upper in ("NULL", "PINCHOUTARRAY", "POR", "PERMI", "PERMJ",
                         "PERMK", "PVCUTOFF", "CPOR", "PRPOR", "GRID"):
                break

            expanded = _expand_cmg_tokens(raw_line)
            all_values.extend(expanded)

            if len(all_values) >= total_expected:
                break

    if len(all_values) < total_expected:
        raise ValueError(
            f"CORNERS block has {len(all_values)} values, "
            f"expected {total_expected} (3 * {vals_per_block})"
        )

    all_values = all_values[:total_expected]

    data = np.array(all_values, dtype=np.float64)
    x_flat = data[:vals_per_block]
    y_flat = data[vals_per_block : 2 * vals_per_block]
    z_flat = data[2 * vals_per_block :]

    shape_block = (2 * nk, 2 * nj, 2 * ni)
    X = x_flat.reshape(shape_block)
    Y = y_flat.reshape(shape_block)
    Z = z_flat.reshape(shape_block)

    corners = np.empty((ni, nj, nk, 8, 3), dtype=np.float64)

    ii = np.arange(ni)
    jj = np.arange(nj)
    kk = np.arange(nk)

    i2 = 2 * ii
    j2 = 2 * jj
    k2 = 2 * kk

    for c_idx, (di, dj, dk) in enumerate([
        (0, 0, 0),  # 0: left-front-bottom
        (1, 0, 0),  # 1: right-front-bottom
        (0, 1, 0),  # 2: left-back-bottom
        (1, 1, 0),  # 3: right-back-bottom
        (0, 0, 1),  # 4: left-front-top
        (1, 0, 1),  # 5: right-front-top
        (0, 1, 1),  # 6: left-back-top
        (1, 1, 1),  # 7: right-back-top
    ]):
        ki = k2[:, None, None] + dk
        ji = j2[None, :, None] + dj
        ix = i2[None, None, :] + di

        ki_bc = np.broadcast_to(ki, (nk, nj, ni))
        ji_bc = np.broadcast_to(ji, (nk, nj, ni))
        ix_bc = np.broadcast_to(ix, (nk, nj, ni))

        corners[:, :, :, c_idx, 0] = X[ki_bc, ji_bc, ix_bc].transpose(2, 1, 0)
        corners[:, :, :, c_idx, 1] = Y[ki_bc, ji_bc, ix_bc].transpose(2, 1, 0)
        corners[:, :, :, c_idx, 2] = Z[ki_bc, ji_bc, ix_bc].transpose(2, 1, 0)

    return corners


def load_or_parse_corners(
    dat_path: str,
    cache_path: Optional[str] = None,
    ni: int = 34,
    nj: int = 16,
    nk: int = 25,
) -> np.ndarray:
    """Load corners from cache, or parse from ``.dat`` and write cache.

    Parameters
    ----------
    dat_path : str
        Path to the CMG ``.dat`` file.
    cache_path : str, optional
        Path for the ``.npy`` cache file.  If ``None``, defaults to
        ``<dat_dir>/.corner_point_cache.npy``.
    ni, nj, nk : int
        Grid dimensions.

    Returns
    -------
    corners : np.ndarray, shape ``(ni, nj, nk, 8, 3)``
    """
    dat = Path(dat_path)
    if cache_path is None:
        cache = dat.parent / ".corner_point_cache.npy"
    else:
        cache = Path(cache_path)

    if cache.exists() and cache.stat().st_mtime >= dat.stat().st_mtime:
        corners = np.load(str(cache))
        if corners.shape == (ni, nj, nk, 8, 3):
            return corners

    corners = parse_corners_from_file(str(dat), ni, nj, nk)
    try:
        np.save(str(cache), corners)
    except OSError:
        pass
    return corners


def get_grid_dimensions(dat_path: str) -> Tuple[int, int, int]:
    """Read ``GRID CORNER NI NJ NK`` from the ``.dat`` file header."""
    with open(dat_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("GRID") and "CORNER" in line:
                parts = line.split()
                idx = parts.index("CORNER")
                return int(parts[idx + 1]), int(parts[idx + 2]), int(parts[idx + 3])
    raise ValueError(f"No 'GRID CORNER' declaration found in {dat_path}")
