import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics as met
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
import copy

def train_epoch(model, loader, optimizer):
    """Performs one training epoch."""
    model.train()
    loss_fn = BCEWithLogitsLoss()
    losses = []

    for batch in loader:
        # Skip tiny batches that could cause BatchNorm issues
        if batch.cochains[0].x.size(0) <= 1:
            continue
            
        optimizer.zero_grad()
        pred = model(batch)
        targets = batch.y.to(torch.float32).view(pred.shape)
        
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.detach().item())
        
    return np.mean(losses)

def evaluate(model, loader):
    """Evaluates model on a data loader."""
    model.eval()
    loss_fn = BCEWithLogitsLoss()
    y_true = []
    y_pred = []
    losses = []
    
    for batch in loader:
        with torch.no_grad():
            pred = model(batch)
            targets = batch.y.to(torch.float32).view(pred.shape)
            
            loss = loss_fn(pred, targets)
            losses.append(loss.detach().item())
            
            # Convert predictions to binary (0 or 1) for accuracy
            binary_pred = (pred > 0).float()
            y_true.append(targets)
            y_pred.append(binary_pred)
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    # Calculate accuracy
    correct = (y_true == y_pred).sum().item()
    total = len(y_true)
    accuracy = correct / total
    mean_loss = np.mean(losses)
    
    return accuracy, mean_loss

def train(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    scheduler_step_size: int = 20,
    scheduler_gamma: float = 0.5,
    early_stop_lr: float = 1e-5,
    train_eval_period: int = 10,
    minimize_metric: bool = False,
    window_size: int = 5  # Should be odd number
):
    """Main training loop with enhanced tracking."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    best_val_score = float('-inf') if not minimize_metric else float('inf')
    best_model = None
    window_center = window_size // 2  # e.g., 2 for window_size=5
    
    # Only store states up to window_center + 1 epochs back
    recent_states = []
    
    train_curve = []
    valid_curve = []
    test_curve = []
    train_loss_curve = []
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer)
        train_loss_curve.append(train_loss)
        
        # Evaluation
        train_score = evaluate(model, train_loader)[0]
        train_curve.append(train_score)
        
        val_score, val_loss = evaluate(model, val_loader)
        valid_curve.append(val_score)
        
        if test_loader is not None:
            test_score, test_loss = evaluate(model, test_loader)
            test_curve.append(test_score)
            
        # Store current model state
        current_state = {
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'scheduler_state_dict': copy.deepcopy(scheduler.state_dict()),
            'val_score': val_score,
        }
        recent_states.append(current_state)
        
        # Keep only window_center + 1 most recent states
        if len(recent_states) > window_center + 1:
            recent_states.pop(0)
            
        # Calculate moving average of validation scores
        if len(valid_curve) >= window_size:
            moving_avg = np.mean(valid_curve[-window_size:])
            
            # Update best model based on moving average
            if moving_avg > best_val_score and epoch >= window_size:
                best_val_score = moving_avg
                best_val_epoch = epoch - window_center
                # Use the oldest state in our recent states (which is window_center epochs back)
                best_model = recent_states[0]
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
            
        # Logging
        if epoch % train_eval_period == 0:
            print(f"\nEpoch {epoch:03d}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc: {val_score:.4f}")
            if test_loader is not None:
                print(f"Test Acc: {test_score:.4f}")
            print(f"LR: {current_lr:.6f}")
            if len(valid_curve) >= window_size:
                print(f"Val Moving Avg: {moving_avg:.4f}")
            if best_model is not None:
                print(f"Best Val Moving Avg: {best_val_score:.4f} (epoch {best_val_epoch})")
    
    # Store training curves
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'best_epoch': best_val_epoch if best_model is not None else -1
    }
    
    return best_model, curves

def test(model, loader, best_model=None):
    """Evaluate on test set."""
    if best_model is not None:
        model.load_state_dict(best_model['model_state_dict'])
    ap_score, test_loss = evaluate(model, loader)
    print(f'Test AP: {ap_score:.4f}')
    return ap_score 