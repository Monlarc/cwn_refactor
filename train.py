import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics as met
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR

def train_epoch(model, loader, optimizer):
    """Performs one training epoch."""
    model.train()
    loss_fn = BCEWithLogitsLoss()
    losses = []

    for batch in tqdm(loader, desc="Training"):
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
    
    for batch in tqdm(loader, desc="Evaluating"):
        with torch.no_grad():
            pred = model(batch)
            targets = batch.y.to(torch.float32).view(pred.shape)
            
            loss = loss_fn(pred, targets)
            losses.append(loss.detach().item())
            
            y_true.append(targets)
            y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    # Calculate metrics
    ap_score = met.average_precision_score(y_true, y_pred)
    mean_loss = np.mean(losses)
    
    return ap_score, mean_loss

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
    minimize_metric: bool = False
):
    """Main training loop with enhanced tracking."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    best_val_epoch = 0
    best_val_score = float('-inf') if not minimize_metric else float('inf')
    best_model = None
    
    train_curve = []
    valid_curve = []
    test_curve = []
    train_loss_curve = []
    params = []
    
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
            
        # Track best model
        is_best = (val_score > best_val_score) if not minimize_metric else (val_score < best_val_score)
        if is_best:
            best_val_score = val_score
            best_val_epoch = epoch
            best_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_score': val_score,
            }
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Early stopping check
        # if current_lr < early_stop_lr:
        #     print("\n!! The minimum learning rate has been reached.")
        #     break
            
        # Parameter change tracking
        # if epoch % train_eval_period == 0:
        #     track_parameter_changes(model, params)
            
        # Logging
        if epoch % train_eval_period == 0:
            print(f"\nEpoch {epoch:03d}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val AP: {val_score:.4f}")
            if test_loader is not None:
                print(f"Test AP: {test_score:.4f}")
            print(f"LR: {current_lr:.6f}")
            print(f"Best Val: {best_val_score:.4f} (epoch {best_val_epoch})")
    
    # Store training curves
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'best_epoch': best_val_epoch
    }
    
    return best_model, curves

def test(model, loader, best_model=None):
    """Evaluate on test set."""
    if best_model is not None:
        model.load_state_dict(best_model['model_state_dict'])
    ap_score, test_loss = evaluate(model, loader)
    print(f'Test AP: {ap_score:.4f}')
    return ap_score 