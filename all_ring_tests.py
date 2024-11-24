import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
from joblib import delayed
import numpy as np
from torch import Tensor
from typing import Union, Optional, List, Dict

from utils import (ProgressParallel, compute_ring_2complex, pyg_to_simplex_tree, 
                  build_tables, extract_boundaries_and_coboundaries_with_rings,
                  build_adj, construct_features, extract_labels, generate_cochain,
                  Complex)


def get_all_cycles(edge_index, max_ring_size=None):
    """Find all cycles in the graph using Johnson's algorithm."""
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    
    # Convert to NetworkX graph
    edge_list = edge_index.T
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Find all simple cycles
    cycles = list(nx.simple_cycles(G))
    
    # Filter by size if specified
    if max_ring_size:
        cycles = [c for c in cycles if len(c) <= max_ring_size]
    
    return [tuple(cycle) for cycle in cycles]

def build_tables_with_all_cycles(edge_index, simplex_tree, size, max_k):
    """Build tables including all cycles, not just induced ones."""
    # Build tables up to edges using existing function
    cell_tables, id_maps = build_tables(simplex_tree, size)
    
    # Find all cycles
    cycles = get_all_cycles(edge_index, max_k)
    
    if len(cycles) > 0:
        # Add cycles as 2-cells
        id_maps += [{}]
        cell_tables += [[]]
        for cycle in cycles:
            next_id = len(cell_tables[2])
            id_maps[2][cycle] = next_id
            cell_tables[2].append(list(cycle))
    
    return cell_tables, id_maps

def process_edge_attributes(edge_attr: Tensor, edge_index: Tensor, id_maps: List[Dict]) -> Tensor:
    """Process edge attributes to create feature matrix for 1-cells."""
    # If edge_attr is a list of scalar features, make it a matrix
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)
    
    # Retrieve feats and check edge features are undirected
    ex = dict()
    for e, edge in enumerate(edge_index.numpy().T):
        canon_edge = tuple(sorted(edge))
        edge_id = id_maps[1][canon_edge]
        edge_feats = edge_attr[e]
        if edge_id in ex:
            assert torch.equal(ex[edge_id], edge_feats)
        else:
            ex[edge_id] = edge_feats

    # Build edge feature matrix
    max_id = max(ex.keys())
    edge_feats = []
    assert len(id_maps[1]) == max_id + 1
    for id in range(max_id + 1):
        edge_feats.append(ex[id])
    edge_features = torch.stack(edge_feats, dim=0)
    
    assert edge_features.dim() == 2
    assert edge_features.size(0) == len(id_maps[1])
    assert edge_features.size(1) == edge_attr.size(1)
    
    return edge_features

def compute_all_cycle_complex(x: Union[Tensor, np.ndarray], 
                            edge_index: Union[Tensor, np.ndarray],
                            edge_attr: Optional[Union[Tensor, np.ndarray]],
                            size: int, 
                            y: Optional[Union[Tensor, np.ndarray]] = None, 
                            max_k: int = 7,
                            include_down_adj=True, 
                            init_method: str = 'sum',
                            init_edges=True, 
                            init_rings=False) -> Complex:
    """Modified version of compute_ring_2complex that uses all cycles."""
    
    # Convert inputs to tensors if needed
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # Create simplicial complex up to edges
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    
    # Build tables with all cycles instead of just induced ones
    cell_tables, id_maps = build_tables_with_all_cycles(edge_index, simplex_tree, size, max_k)
    complex_dim = len(id_maps)-1
    
    # Use existing functions for the rest of the pipeline
    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(
        simplex_tree, id_maps)
    
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(
        boundaries, co_boundaries, id_maps, complex_dim, include_down_adj)
    
    # Rest of the processing is identical to compute_ring_2complex
    xs = [x, None, None]
    constructed_features = construct_features(x, cell_tables, init_method)
    
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]
    
    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        else:
            xs[1] = process_edge_attributes(edge_attr, edge_index, id_maps)
    
    v_y, complex_y = extract_labels(y, size)
    
    cochains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, 
                                 shared_boundaries, shared_coboundaries,
                                 cell_tables, boundaries_tables, 
                                 complex_dim=complex_dim, y=y)
        cochains.append(cochain)
    
    return Complex(*cochains, y=complex_y, dimension=complex_dim)

def convert_graph_dataset_with_all_cycles(dataset, max_ring_size=7, 
                                        include_down_adj=False,
                                        init_method: str = 'sum', 
                                        init_edges=True, 
                                        init_rings=False,
                                        n_jobs=1):
    """Convert a graph dataset to cell complexes using all cycles."""
    dimension = -1
    num_features = [None, None, None]
    
    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x
    
    # Process dataset in parallel using the same structure as the original
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    complexes = parallel(delayed(compute_all_cycle_complex)(
        maybe_convert_to_numpy(data.x), 
        maybe_convert_to_numpy(data.edge_index),
        maybe_convert_to_numpy(data.edge_attr),
        data.num_nodes, 
        y=maybe_convert_to_numpy(data.y), 
        max_k=max_ring_size,
        include_down_adj=include_down_adj, 
        init_method=init_method,
        init_edges=init_edges, 
        init_rings=init_rings) for data in dataset)
    
    # Validation code is identical to original
    for c, complex in enumerate(complexes):
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features
        
        graph = dataset[c]
        if complex.y is None:
            assert graph.y is None
        else:
            assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.cochains[0].x, graph.x)
        if complex.dimension >= 1:
            assert complex.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)
    
    return complexes, dimension, num_features[:dimension+1]

def test_all_cycles():
    """Test the all-cycles implementation with MUTAG."""
    
    # Test case 3: Real dataset example
    from torch_geometric.datasets import TUDataset
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    
    print("\nAnalyzing MUTAG Dataset:")
    print("=" * 50)
    
    total_all_cycles = 0
    total_induced_cycles = 0
    molecules_with_difference = 0
    
    for i in range(len(dataset)):
        complex_all = compute_all_cycle_complex(
            dataset[i].x, dataset[i].edge_index, dataset[i].edge_attr, 
            size=dataset[i].num_nodes, max_k=7,
            init_edges=True, init_rings=True
        )
        
        complex_induced = compute_ring_2complex(
            dataset[i].x, dataset[i].edge_index, dataset[i].edge_attr, 
            size=dataset[i].num_nodes, max_k=7,
            init_edges=True, init_rings=True
        )
        
        all_cycles = len(complex_all.cochains[2].x)
        induced_cycles = len(complex_induced.cochains[2].x)
        
        total_all_cycles += all_cycles
        total_induced_cycles += induced_cycles
        
        if all_cycles > induced_cycles:
            molecules_with_difference += 1
            print(f"\nMolecule {i}:")
            print(f"Number of nodes: {dataset[i].num_nodes}")
            print(f"Number of edges: {dataset[i].edge_index.size(1)//2}")
            print(f"All cycles found: {all_cycles}")
            print(f"Induced cycles found: {induced_cycles}")
            
            # Print the actual cycles
            cycles = get_all_cycles(dataset[i].edge_index)
            print("\nAll cycles found:")
            for c in cycles:
                print(f"Cycle: {c}")
    
    print("\nSummary Statistics:")
    print(f"Total molecules: {len(dataset)}")
    print(f"Molecules with extra cycles: {molecules_with_difference}")
    print(f"Total all cycles found: {total_all_cycles}")
    print(f"Total induced cycles found: {total_induced_cycles}")
    print(f"Additional cycles found: {total_all_cycles - total_induced_cycles}")
    
    # Timing test
    import time
    print("\nTiming Test on MUTAG Dataset")
    num_graphs = 100  # Test first 100 graphs
    
    start = time.time()
    for i in range(num_graphs):
        data = dataset[i]
        complex_all = compute_all_cycle_complex(
            data.x, data.edge_index, data.edge_attr, 
            size=data.num_nodes, max_k=7,
            init_edges=True, init_rings=True
        )
    all_cycles_time = time.time() - start
    
    start = time.time()
    for i in range(num_graphs):
        data = dataset[i]
        complex_induced = compute_ring_2complex(
            data.x, data.edge_index, data.edge_attr, 
            size=data.num_nodes, max_k=7,
            init_edges=True, init_rings=True
        )
    induced_cycles_time = time.time() - start
    
    print(f"All cycles time: {all_cycles_time:.2f}s")
    print(f"Induced cycles time: {induced_cycles_time:.2f}s")
    print(f"Ratio: {all_cycles_time/induced_cycles_time:.2f}x slower")

def test_simple_graphs():
    """Test cycle detection on simple graphs with known cycle counts."""
    test_cases = {
        "Triangle": {
            "edge_index": torch.tensor([[0, 1, 1, 2, 2, 0],
                                      [1, 0, 2, 1, 0, 2]]).long(),
            "size": 3,
            "expected_induced": 1,
            "expected_all": 1,
            "description": "Simple triangle - should have exactly one cycle"
        },
        "Square": {
            "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                                      [1, 0, 2, 1, 3, 2, 0, 3]]).long(),
            "size": 4,
            "expected_induced": 1,
            "expected_all": 1,
            "description": "Simple square - should have exactly one cycle"
        },
        "Square with Diagonal": {
            "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 1, 3],
                                      [1, 0, 2, 1, 3, 2, 0, 3, 3, 1]]).long(),
            "size": 4,
            "expected_induced": 2,
            "expected_all": 3,
            "description": "Square with diagonal - should have 3 total cycles (2 triangles + 1 square) but only 2 induced cycles"
        },
        "Bow Tie": {
            "edge_index": torch.tensor([[0, 1, 1, 2, 0, 2, 2, 3, 3, 4, 2, 4],
                                      [1, 0, 2, 1, 2, 0, 3, 2, 4, 3, 4, 2]]).long(),
            "size": 5,
            "expected_induced": 2,
            "expected_all": 2,
            "description": "Two triangles sharing a vertex - should have exactly 2 cycles"
        },
        "Complete Graph K4": {
            "edge_index": torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                      [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]]).long(),
            "size": 4,
            "expected_induced": 4,
            "expected_all": 7,
            "description": "Complete graph on 4 vertices - should have 4 induced cycles but 7 total cycles"
        },
        "Pentagon": {
            "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                                      [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]]).long(),
            "size": 5,
            "expected_induced": 1,
            "expected_all": 1,
            "description": "Simple pentagon - should have exactly one cycle"
        }
    }
    
    print("\nRunning Simple Graph Tests:")
    print("=" * 50)
    
    for name, case in test_cases.items():
        print(f"\nTesting {name}:")
        print(case["description"])
        
        # Create dummy features
        x = torch.ones(case["size"], 2)  # 2 features per node
        edge_attr = torch.ones(case["edge_index"].size(1), 2)  # 2 features per edge
        
        # Process with both methods
        complex_all = compute_all_cycle_complex(
            x, case["edge_index"], edge_attr, 
            size=case["size"], max_k=7,
            init_edges=True, init_rings=True
        )
        
        from utils import compute_ring_2complex
        complex_induced = compute_ring_2complex(
            x, case["edge_index"], edge_attr, 
            size=case["size"], max_k=7,
            init_edges=True, init_rings=True
        )
        
        # Get results
        all_cycles = len(complex_all.cochains[2].x)
        induced_cycles = len(complex_induced.cochains[2].x)
        
        # Print and verify results
        print(f"Expected induced cycles: {case['expected_induced']}")
        print(f"Found induced cycles: {induced_cycles}")
        print(f"Expected all cycles: {case['expected_all']}")
        print(f"Found all cycles: {all_cycles}")
        
        try:
            assert induced_cycles == case["expected_induced"]
            assert all_cycles == case["expected_all"]
            print("✓ Test passed!")
        except AssertionError:
            print("✗ Test failed!")
            print(f"Induced cycles mismatch: expected {case['expected_induced']}, got {induced_cycles}")
            print(f"All cycles mismatch: expected {case['expected_all']}, got {all_cycles}")

def main():
    # Run simple graph tests
    test_simple_graphs()
    
    # Run the original tests
    test_all_cycles()

if __name__ == "__main__":
    main()