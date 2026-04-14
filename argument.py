import argparse
import torch

def get_args(args_list=None):
    parser = argparse.ArgumentParser(description='HSG improved')

    parser.add_argument('--root_path', type=str, default='./data/')
    parser.add_argument('--data_name', type=str)

    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)

    parser.add_argument('--signal_len', type=int, default=16)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--s', type=float, default=2.0)
    parser.add_argument('--frame_type', type=str, default='Haar', choices=['Haar', 'Linear'])
    parser.add_argument('--cheb_order', type=int, default=2)
    parser.add_argument('--lev', type=int, default=1)
    parser.add_argument('--ma_kernel', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()