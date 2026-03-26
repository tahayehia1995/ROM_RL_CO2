"""
Graph Construction for GNN E2C
================================
Builds PyTorch Geometric graphs from the structured 3D reservoir grid.
Handles multiple geological realizations with different inactive cell masks,
permeability fields, and porosity fields.

Each realization produces a distinct graph with:
- Nodes: one per active (non-masked) grid cell
- Edges: 6-connected face neighbors between active cells
- Edge attributes: harmonic-mean transmissibility + direction one-hot
- Static node features: PERMI, POROS, well indicators (fixed per realization)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


NEIGHBOR_OFFSETS = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
]

DIRECTION_VECTORS = torch.tensor([
    [1, 0, 0, 0, 0, 0],  # -x
    [0, 1, 0, 0, 0, 0],  # +x
    [0, 0, 1, 0, 0, 0],  # -y
    [0, 0, 0, 1, 0, 0],  # +y
    [0, 0, 0, 0, 1, 0],  # -z
    [0, 0, 0, 0, 0, 1],  # +z
], dtype=torch.float32)


@dataclass
class GraphStructure:
    """Precomputed graph data for a single geological realization."""
    edge_index: torch.Tensor           # (2, E) int64
    edge_attr: torch.Tensor            # (E, 7) float32 — transmissibility + direction
    static_node_features: torch.Tensor # (N_active, F_static) float32
    node_to_grid: torch.Tensor         # (N_active, 3) int64 — (i,j,k) per node
    grid_to_node: torch.Tensor         # (Nx, Ny, Nz) int64 — node id or -1
    node_positions: torch.Tensor       # (N_active, 3) float32 — normalized coords
    num_active_nodes: int
    active_mask: torch.Tensor          # (Nx, Ny, Nz) bool
    well_node_indices: Dict = field(default_factory=dict)
    well_indicator_features: Optional[torch.Tensor] = None


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def build_graph_from_grid(
    active_mask: np.ndarray,
    permi: np.ndarray,
    poros: np.ndarray,
    well_locations: Dict,
    grid_shape: Tuple[int, int, int],
) -> GraphStructure:
    """
    Build a graph from a single realization's data.

    Args:
        active_mask: (Nx, Ny, Nz) bool array
        permi: (Nx, Ny, Nz) raw permeability values (NOT normalized)
        poros: (Nx, Ny, Nz) raw porosity values (NOT normalized)
        well_locations: dict with 'injectors' and 'producers' sub-dicts
        grid_shape: (Nx, Ny, Nz) tuple
    """
    Nx, Ny, Nz = grid_shape

    active_indices = np.argwhere(active_mask)  # (N_active, 3)
    N_active = len(active_indices)

    grid_to_node = np.full((Nx, Ny, Nz), -1, dtype=np.int64)
    for node_id, (i, j, k) in enumerate(active_indices):
        grid_to_node[i, j, k] = node_id

    # --- Build edges ---
    src_list, dst_list = [], []
    edge_attr_list = []

    for node_id, (i, j, k) in enumerate(active_indices):
        for dir_idx, (di, dj, dk) in enumerate(NEIGHBOR_OFFSETS):
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                neighbor_node = grid_to_node[ni, nj, nk]
                if neighbor_node >= 0:
                    src_list.append(node_id)
                    dst_list.append(neighbor_node)
                    trans = _harmonic_mean(
                        float(permi[i, j, k]),
                        float(permi[ni, nj, nk]),
                    )
                    direction = DIRECTION_VECTORS[dir_idx].numpy()
                    edge_attr_list.append(
                        np.concatenate([[trans], direction])
                    )

    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float32)

    # --- Normalize transmissibility channel ---
    if edge_attr.shape[0] > 0:
        trans_col = edge_attr[:, 0]
        t_max = trans_col.max()
        if t_max > 0:
            edge_attr[:, 0] = trans_col / t_max

    # --- Static node features ---
    # Collect well info
    all_injectors = well_locations.get('injectors', {})
    all_producers = well_locations.get('producers', {})
    num_wells = len(all_injectors) + len(all_producers)

    static_feats = []
    for node_id, (i, j, k) in enumerate(active_indices):
        perm_val = float(permi[i, j, k])
        poro_val = float(poros[i, j, k])

        is_inj = 0.0
        is_prod = 0.0
        well_id = np.zeros(max(num_wells, 1), dtype=np.float32)

        well_idx = 0
        for wname, coords in sorted(all_injectors.items()):
            wx, wy = coords[0], coords[1]
            if i == wx and j == wy:
                is_inj = 1.0
                well_id[well_idx] = 1.0
            well_idx += 1
        for wname, coords in sorted(all_producers.items()):
            wx, wy = coords[0], coords[1]
            if i == wx and j == wy:
                is_prod = 1.0
                well_id[well_idx] = 1.0
            well_idx += 1

        feat = np.concatenate([
            [perm_val, poro_val, is_inj, is_prod],
            well_id,
        ])
        static_feats.append(feat)

    static_node_features = torch.tensor(
        np.array(static_feats), dtype=torch.float32
    )

    # Normalize PERMI and POROS columns within this realization
    for col in [0, 1]:
        col_data = static_node_features[:, col]
        c_min, c_max = col_data.min(), col_data.max()
        if c_max > c_min:
            static_node_features[:, col] = (col_data - c_min) / (c_max - c_min)
        else:
            static_node_features[:, col] = 0.0

    # --- Node positions (normalized 0-1) ---
    node_positions = torch.tensor(active_indices, dtype=torch.float32)
    node_positions[:, 0] /= max(Nx - 1, 1)
    node_positions[:, 1] /= max(Ny - 1, 1)
    node_positions[:, 2] /= max(Nz - 1, 1)

    # --- Well-node index mapping ---
    # For each well, find all node indices where (i,j) matches the well's
    # (x,y) location across all k layers.  Wells penetrate all layers.
    # Order: injectors sorted by name, then producers sorted by name.
    well_node_indices = {}
    grid_to_node_np = grid_to_node

    well_order = []
    for wname in sorted(all_injectors.keys()):
        well_order.append(('inj', wname, all_injectors[wname]))
    for wname in sorted(all_producers.keys()):
        well_order.append(('prod', wname, all_producers[wname]))

    for wtype, wname, coords in well_order:
        wx, wy = coords[0], coords[1]
        node_ids = []
        for kk in range(Nz):
            nid = grid_to_node_np[wx, wy, kk]
            if nid >= 0:
                node_ids.append(int(nid))
        well_node_indices[wname] = {
            'type': wtype,
            'node_ids': node_ids,
        }

    # --- Per-node well indicator features for dynamic encoder ---
    # Shape: (N_active, 2 + num_wells + 1)
    # [is_inj, is_prod, well_I1, well_I2, ..., well_P1, ..., control_placeholder]
    # The control_placeholder column is filled at runtime with actual values.
    n_well_indicator_cols = 2 + num_wells + 1
    well_indicators = np.zeros((N_active, n_well_indicator_cols), dtype=np.float32)

    well_idx = 0
    for wtype, wname, coords in well_order:
        for nid in well_node_indices[wname]['node_ids']:
            if wtype == 'inj':
                well_indicators[nid, 0] = 1.0  # is_inj
            else:
                well_indicators[nid, 1] = 1.0  # is_prod
            well_indicators[nid, 2 + well_idx] = 1.0  # well one-hot
        well_idx += 1

    return GraphStructure(
        edge_index=edge_index,
        edge_attr=edge_attr,
        static_node_features=static_node_features,
        node_to_grid=torch.tensor(active_indices, dtype=torch.long),
        grid_to_node=torch.tensor(grid_to_node, dtype=torch.long),
        node_positions=node_positions,
        num_active_nodes=N_active,
        active_mask=torch.tensor(active_mask, dtype=torch.bool),
        well_node_indices=well_node_indices,
        well_indicator_features=torch.tensor(well_indicators, dtype=torch.float32),
    )


class GraphManager:
    """
    Manages per-realization graph structures for the GNN E2C model.

    At init time:
      1. Loads the 4D inactive mask and raw PERMI / POROS from H5 files.
      2. Groups cases into unique realizations (by comparing masks).
      3. Builds one GraphStructure per unique realization.

    At training time:
      - ``create_pyg_batch()`` converts a standard batch of grid tensors
        into a PyG Batch, using ``case_indices`` to select the correct
        graph structure per sample.
    """

    def __init__(
        self,
        mask_path: str,
        spatial_data_dir: str,
        well_locations: Dict,
        grid_shape: Tuple[int, int, int],
        device: torch.device = torch.device('cpu'),
        verbose: bool = False,
    ):
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "GraphManager requires torch_geometric. "
                "Install with: pip install torch_geometric"
            )

        self.device = device
        self.grid_shape = grid_shape
        Nx, Ny, Nz = grid_shape

        # --- Load mask ---
        with h5py.File(mask_path, 'r') as f:
            raw_mask = f['active_mask'][...]

        if raw_mask.ndim == 3:
            raw_mask = raw_mask[np.newaxis, ...]  # (1, Nx, Ny, Nz)
        raw_mask = raw_mask.astype(bool)
        num_cases = raw_mask.shape[0]

        # --- Load raw PERMI and POROS (first timestep, before normalization) ---
        permi_path = os.path.join(spatial_data_dir, 'batch_spatial_properties_PERMI.h5')
        poros_path = os.path.join(spatial_data_dir, 'batch_spatial_properties_POROS.h5')

        with h5py.File(permi_path, 'r') as f:
            permi_all = f['data'][:, 0, :, :, :]  # (cases, Nx, Ny, Nz) first timestep

        with h5py.File(poros_path, 'r') as f:
            poros_all = f['data'][:, 0, :, :, :]

        # --- Identify unique realizations ---
        realization_ids, self.case_to_realization = self._identify_realizations(
            raw_mask, permi_all, num_cases
        )
        self.num_realizations = len(realization_ids)

        if verbose:
            print(f"GNN GraphManager: {num_cases} cases -> "
                  f"{self.num_realizations} unique realizations")

        # --- Build graph per realization ---
        self.graphs: List[GraphStructure] = []
        for r_idx, case_idx in enumerate(realization_ids):
            mask_r = raw_mask[case_idx]
            permi_r = permi_all[case_idx]
            poros_r = poros_all[case_idx]

            gs = build_graph_from_grid(
                mask_r, permi_r, poros_r, well_locations, grid_shape
            )
            # Move graph tensors to device
            gs.edge_index = gs.edge_index.to(device)
            gs.edge_attr = gs.edge_attr.to(device)
            gs.static_node_features = gs.static_node_features.to(device)
            gs.node_to_grid = gs.node_to_grid.to(device)
            gs.grid_to_node = gs.grid_to_node.to(device)
            gs.node_positions = gs.node_positions.to(device)
            gs.active_mask = gs.active_mask.to(device)
            if gs.well_indicator_features is not None:
                gs.well_indicator_features = gs.well_indicator_features.to(device)
            self.graphs.append(gs)

            if verbose:
                print(f"  Realization {r_idx}: {gs.num_active_nodes} active nodes, "
                      f"{gs.edge_index.shape[1]} edges")

    @staticmethod
    def _identify_realizations(
        masks: np.ndarray, permi: np.ndarray, num_cases: int
    ) -> Tuple[List[int], np.ndarray]:
        """Group cases by unique (mask, permi) combinations."""
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

    def get_realization(self, case_idx: int) -> int:
        """Map a case index to its realization index."""
        if case_idx < len(self.case_to_realization):
            return int(self.case_to_realization[case_idx])
        return 0

    def get_graph(self, realization_idx: int) -> GraphStructure:
        return self.graphs[min(realization_idx, len(self.graphs) - 1)]

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def extract_dynamic_node_features(
        self,
        grid_tensor: torch.Tensor,
        case_indices: Optional[torch.Tensor],
        channel_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract dynamic node features from grid tensor for a batch.

        Args:
            grid_tensor: (B, C, Nx, Ny, Nz)
            case_indices: (B,) int — maps each sample to original case id
            channel_indices: which channels to extract (e.g. [0,1] for SG,PRES)

        Returns:
            flat_features: (total_nodes_in_batch, len(channel_indices))
            batch_vector: (total_nodes_in_batch,) int — sample assignment
            realization_ids: (B,) int — realization index per sample
        """
        B = grid_tensor.shape[0]
        device = grid_tensor.device

        if case_indices is None:
            real_ids = torch.zeros(B, dtype=torch.long, device=device)
        else:
            real_ids = torch.tensor(
                [self.get_realization(int(ci)) for ci in case_indices],
                dtype=torch.long, device=device,
            )

        all_feats = []
        batch_ids = []

        for b in range(B):
            r = int(real_ids[b])
            gs = self.graphs[r]
            coords = gs.node_to_grid  # (N_active, 3)
            n = gs.num_active_nodes

            feats = grid_tensor[
                b, channel_indices
            ][:, coords[:, 0], coords[:, 1], coords[:, 2]]  # (C_sel, N)
            all_feats.append(feats.t())  # (N, C_sel)
            batch_ids.append(torch.full((n,), b, dtype=torch.long, device=device))

        flat_features = torch.cat(all_feats, dim=0)
        batch_vector = torch.cat(batch_ids, dim=0)
        return flat_features, batch_vector, real_ids

    def get_batched_graph(
        self,
        case_indices: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build batched edge_index, edge_attr, static features, positions,
        and batch vector for a batch of samples.

        Returns:
            edge_index: (2, total_edges)
            edge_attr: (total_edges, 7)
            static_features: (total_nodes, F_static)
            node_positions: (total_nodes, 3)
            batch_vector: (total_nodes,) int
        """
        if case_indices is None:
            real_ids = [0] * batch_size
        else:
            real_ids = [self.get_realization(int(ci)) for ci in case_indices]

        all_edge_src, all_edge_dst = [], []
        all_edge_attr = []
        all_static = []
        all_positions = []
        all_batch = []
        node_offset = 0

        for b in range(batch_size):
            gs = self.graphs[real_ids[b]]
            n = gs.num_active_nodes

            all_edge_src.append(gs.edge_index[0] + node_offset)
            all_edge_dst.append(gs.edge_index[1] + node_offset)
            all_edge_attr.append(gs.edge_attr)
            all_static.append(gs.static_node_features)
            all_positions.append(gs.node_positions)
            all_batch.append(
                torch.full((n,), b, dtype=torch.long, device=device)
            )
            node_offset += n

        edge_index = torch.stack([
            torch.cat(all_edge_src),
            torch.cat(all_edge_dst),
        ], dim=0)
        edge_attr = torch.cat(all_edge_attr, dim=0)
        static_features = torch.cat(all_static, dim=0)
        node_positions = torch.cat(all_positions, dim=0)
        batch_vector = torch.cat(all_batch, dim=0)

        return edge_index, edge_attr, static_features, node_positions, batch_vector

    def scatter_to_grid(
        self,
        node_features: torch.Tensor,
        batch_vector: torch.Tensor,
        case_indices: Optional[torch.Tensor],
        batch_size: int,
        num_channels: int,
    ) -> torch.Tensor:
        """
        Scatter flat node features back to grid tensors.

        Args:
            node_features: (total_nodes, num_channels)
            batch_vector: (total_nodes,)
            case_indices: (B,) int
            batch_size: B
            num_channels: output channels

        Returns:
            grid: (B, num_channels, Nx, Ny, Nz) — zeros at inactive cells
        """
        Nx, Ny, Nz = self.grid_shape
        device = node_features.device
        grid = torch.zeros(
            batch_size, num_channels, Nx, Ny, Nz,
            dtype=node_features.dtype, device=device,
        )

        if case_indices is None:
            real_ids = [0] * batch_size
        else:
            real_ids = [self.get_realization(int(ci)) for ci in case_indices]

        node_offset = 0
        for b in range(batch_size):
            gs = self.graphs[real_ids[b]]
            n = gs.num_active_nodes
            coords = gs.node_to_grid  # (N, 3) on device

            feats = node_features[node_offset:node_offset + n]  # (N, C)
            grid[b, :, coords[:, 0], coords[:, 1], coords[:, 2]] = feats.t()
            node_offset += n

        return grid
