import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argument import get_args
from src.data_provider import data_provider
from src.model import HSG
from src.utils import set_seed, evaluate

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y.permute(0, 2, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in tqdm(loader, desc='Eval', leave=False):
        out = model(x.to(device))
        preds.append(out.cpu().numpy())
        trues.append(y.permute(0, 2, 1).numpy())
    return evaluate(np.concatenate(trues), np.concatenate(preds))

def main(args=None):
    if args is None:
        args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    kw = dict(root_path=args.root_path, dataset_name=args.data_name,
              seq_len=args.seq_len, pred_len=args.pred_len,
              batch_size=args.batch_size)
    train_set, train_loader = data_provider(flag='train', **kw)
    val_set, val_loader = data_provider(flag='val', **kw)
    test_set, test_loader = data_provider(flag='test', **kw)
    num_nodes = train_set.data.shape[1]
    print(f"Dataset: {args.data_name} | N = {num_nodes} | "
          f"T_in={args.seq_len} | T_out={args.pred_len}")
    
    model = HSG(
        seq_len=args.seq_len, pred_len=args.pred_len, signal_len=args.signal_len,
        num_nodes=num_nodes, embed_size=args.embed_size, hidden_size=args.hidden_size,
        k=args.k, s=args.s, frame_type=args.frame_type, cheb_order=args.cheb_order,
        lev=args.lev, ma_kernel=args.ma_kernel
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HSG | Total params: {total_params} | Trainable params: {trainable_params}")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    best_state = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mape, val_mae, val_rmse = eval_model(model, val_loader, device)
        scheduler.step(val_mae)
        
        print(f'Epoch {epoch:03d}/{args.epochs} | '
              f'train_loss {train_loss:.6f} | '
              f'val_MAPE {val_mape:.4f} | val_MAE {val_mae:.6f} | '
              f'val_RMSE {val_rmse:.6f}')
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            print(f"---> best val_MAE = {best_val_mae:.6f} (epoch {epoch})")
            
    model.load_state_dict(best_state)
    test_mape, test_mae, test_rmse = eval_model(model, test_loader, device)
    print('\n' + '=' * 60)
    print('Test Results (best val-MAE checkpoint)')
    print(f'MAPE : {test_mape:.6f}')
    print(f'MAE  : {test_mae:.6f}')
    print(f'RMSE : {test_rmse:.6f}')
    print('=' * 60)
    
    return best_val_mae

if __name__ == '__main__':
    main()