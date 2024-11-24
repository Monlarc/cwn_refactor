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
from gin import EmbedGIN

def set_all_seeds(seed=42):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_dataset(batch_size=32, seed=42):
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
    g.manual_seed(seed)
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

def run_trial(seed, hidden_channels=64, num_layers=4):
    """Run a single trial with given seed."""
    set_all_seeds(seed)
    
    # Process dataset
    train_loader, val_loader, test_loader, num_atom_types, num_classes = process_dataset(seed=seed)
    
    # Train SparseCIN
    cin_model = EmbedSparseCIN(
        atom_types=num_atom_types,
        out_size=1,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        max_dim=2,
        dropout=0.5,
        jump_mode='cat',
        nonlinearity='relu',
        readout='sum',
        final_readout='sum'
    )
    
    best_cin_model, cin_curves = train(
        model=cin_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        learning_rate=0.01,
        scheduler_step_size=20,
        scheduler_gamma=0.5,
        window_size=5
    )
    
    cin_score = test(cin_model, test_loader, best_cin_model)
    
    # Train GIN
    gin_model = EmbedGIN(
        atom_types=num_atom_types,
        bond_types=1,
        out_size=1,
        num_layers=num_layers,
        hidden=hidden_channels,
        dropout_rate=0.5,
        nonlinearity='relu',
        readout='sum',
        train_eps=True,
        apply_dropout_before='lin1'
    )
    
    best_gin_model, gin_curves = train(
        model=gin_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        learning_rate=0.01,
        scheduler_step_size=20,
        scheduler_gamma=0.5,
        window_size=5
    )
    
    gin_score = test(gin_model, test_loader, best_gin_model)
    
    return {
        'cin': {
            'score': cin_score,
            'curves': cin_curves
        },
        'gin': {
            'score': gin_score,
            'curves': gin_curves
        }
    }

def main():
    # Number of trials
    n_trials = 25
    seeds = range(51990, 51990 + n_trials)
    
    
    # Store results for each trial
    all_results = []
    
    # Run trials
    for i, seed in enumerate(seeds):
        print(f"\nTrial {i+1}/{n_trials} (seed={seed})")
        trial_results = run_trial(seed)
        all_results.append(trial_results)
        
        # Print running statistics
        cin_scores = [r['cin']['score'] for r in all_results]
        gin_scores = [r['gin']['score'] for r in all_results]
        
        print("\nCurrent Statistics:")
        print(f"SparseCIN: {np.mean(cin_scores)*100:.1f} ± {np.std(cin_scores)*100:.1f}")
        print(f"GIN: {np.mean(gin_scores)*100:.1f} ± {np.std(gin_scores)*100:.1f}")
    
    # Compute final statistics
    cin_scores = [r['cin']['score'] for r in all_results]
    gin_scores = [r['gin']['score'] for r in all_results]
    
    print("\nFinal Results:")
    print(f"SparseCIN: {np.mean(cin_scores)*100:.1f} ± {np.std(cin_scores)*100:.1f}")
    print(f"GIN: {np.mean(gin_scores)*100:.1f} ± {np.std(gin_scores)*100:.1f}")
    
    # Plot average learning curves
    plot_average_curves(all_results)

def plot_average_curves(all_results):
    """Plot average training curves with standard deviation bands."""
    plt.figure(figsize=(15, 5))
    
    metrics = ['train_loss', 'val', 'test']
    titles = ['Training Loss', 'Validation Accuracy', 'Test Accuracy']
    
    # Calculate mean and std of best validation epochs
    cin_best_epochs = [r['cin']['curves']['best_epoch'] for r in all_results]
    gin_best_epochs = [r['gin']['curves']['best_epoch'] for r in all_results]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, idx+1)
        
        # Get curves for both models
        for model, best_epochs, color in [('cin', cin_best_epochs, 'blue'), 
                                        ('gin', gin_best_epochs, 'orange')]:
            curves = [r[model]['curves'][metric] for r in all_results]
            # Pad sequences to same length if necessary
            max_len = max(len(c) for c in curves)
            curves = [c + [c[-1]]*(max_len - len(c)) for c in curves]
            
            # Convert to numpy array
            curves = np.array(curves)
            mean = np.mean(curves, axis=0)
            std = np.std(curves, axis=0)
            
            # Plot mean and std
            epochs = range(len(mean))
            plt.plot(epochs, mean, label=model.upper(), color=color)
            plt.fill_between(epochs, mean-std, mean+std, alpha=0.2, color=color)
            
            # Plot vertical line for best epoch with std band
            best_epoch_mean = np.mean(best_epochs)
            best_epoch_std = np.std(best_epochs)
            plt.axvline(x=best_epoch_mean, color=color, linestyle='--', 
                       alpha=0.5, label=f'{model.upper()} Best: {best_epoch_mean:.1f}±{best_epoch_std:.1f}')
            # plt.fill_betweenx(plt.ylim(), best_epoch_mean-best_epoch_std, 
            #                 best_epoch_mean+best_epoch_std, color=color, alpha=0.1)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy' if title != 'Training Loss' else 'Loss')
        plt.title(title)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()