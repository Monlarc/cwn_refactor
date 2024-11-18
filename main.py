import torch
from torch_geometric.datasets import TUDataset
from torch.utils.data import DataLoader
from sparse_cin import EmbedSparseCIN
from train import train, test
from utils import convert_graph_dataset_with_rings
from complex import ComplexBatch
import random
import matplotlib.pyplot as plt
import numpy as np
import os

def set_all_seeds(seed=42):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_dataset(batch_size=32):
    """Load and process the MUTAG dataset into cell complexes with rings."""
    # Load MUTAG dataset
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    
    # Convert graphs to cell complexes with rings
    complexes, dimension, num_features = convert_graph_dataset_with_rings(
        dataset,
        max_ring_size=7,
        include_down_adj=True,
        init_method='sum',
        init_edges=True,
        init_rings=True,
        n_jobs=4  # Adjust based on your CPU
    )
    
    # Print class distribution of full dataset
    labels = [c.y.item() for c in complexes]
    print("\nFull dataset:")
    print(f"Total samples: {len(labels)}")
    print(f"Class 0: {labels.count(0)}, Class 1: {labels.count(1)}")
    
    # Split dataset with fixed seed
    # Ensure reproducibility
    g = torch.Generator()
    g.manual_seed(42)
    indices = torch.randperm(len(complexes), generator=g)
    
    train_size = int(0.8 * len(complexes))
    val_size = int(0.1 * len(complexes))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_dataset = [complexes[i] for i in train_indices]
    val_dataset = [complexes[i] for i in val_indices]
    test_dataset = [complexes[i] for i in test_indices]
    
    # Print split statistics
    print("\nSplit statistics:")
    for name, split in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        split_labels = [c.y.item() for c in split]
        print(f"\n{name} set:")
        print(f"Size: {len(split_labels)}")
        print(f"Class 0: {split_labels.count(0)}, Class 1: {split_labels.count(1)}")
        print(f"Class balance: {split_labels.count(1)/len(split_labels):.2f}")
    
    # batch_size = 1
    # Create data loaders with ComplexBatch
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=g,
        collate_fn=ComplexBatch.from_complex_list
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=ComplexBatch.from_complex_list
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=ComplexBatch.from_complex_list
    )
    
    return (train_loader, val_loader, test_loader, 
            dataset.num_node_features, dataset.num_classes)

def plot_training_curves(curves, save_path=None):
    """Plot training, validation and test curves."""
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(curves['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot AP scores
    plt.subplot(1, 2, 2)
    epochs = range(len(curves['train_loss']))  # Use all epochs
    if curves['train']:
        plt.plot(epochs, curves['train'], label='Train AP', marker='o')
    plt.plot(epochs, curves['val'], label='Validation AP', marker='o')
    if curves['test']:
        plt.plot(epochs, curves['test'], label='Test AP', marker='o')
    plt.axvline(x=curves['best_epoch'], color='r', linestyle='--', label='Best Val Epoch')
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.title('Model Performance')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Set seeds first thing
    set_all_seeds(42)
    
    # Process dataset
    train_loader, val_loader, test_loader, num_atom_types, num_classes = process_dataset()
    
    # print(f'Number of atom types: {num_atom_types}')
    # print(f'Number of classes: {num_classes}')
    
    # Initialize model
    model = EmbedSparseCIN(
        atom_types=num_atom_types,
        out_size=1,
        hidden_channels=64,
        num_layers=4,
        max_dim=2,  # 0: atoms, 1: bonds, 2: rings
        dropout=0.5,
        jump_mode='cat',
        nonlinearity='relu',
        readout='sum',
        final_readout='sum'
    )
    
    # Training
    best_model, curves = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        learning_rate=0.01,
        scheduler_step_size=20,
        scheduler_gamma=0.5
    )
    
    # Final evaluation
    final_score = test(model, test_loader, best_model)
    print(f'Final test score: {final_score:.4f}')
    
    # Plot training curves
    plot_training_curves(curves)

if __name__ == "__main__":
    main()