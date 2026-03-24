import argparse
import torch

def get_args(args_list=None):
    parser = argparse.ArgumentParser(description='FreqTimeHPG — improved')

    # ---- data ----
    parser.add_argument('--root_path',  type=str, default='./data/')
    parser.add_argument('--data_name',  type=str)

    # ---- sequence ----
    parser.add_argument('--seq_len',  type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)

    # ---- model ----
    parser.add_argument('--signal_len',  type=int,   default=16)
    parser.add_argument('--embed_size',  type=int,   default=128)
    parser.add_argument('--hidden_size', type=int,   default=64)
    parser.add_argument('--k',           type=int,   default=2)
    parser.add_argument('--s',           type=float, default=2.0)
    parser.add_argument('--frame_type',  type=str,   default='Haar',
                        choices=['Haar', 'Linear'])
    parser.add_argument('--cheb_order',  type=int,   default=2)
    parser.add_argument('--lev',         type=int,   default=1)
    parser.add_argument('--ma_kernel',   type=int,   default=3)

    # ---- unused flags kept only for backwards CLI compatibility ----
    # These are accepted but have no effect on the model.
    # They exist so old sweep scripts do not break.
    parser.add_argument('--learnable_filters', action='store_true', default=False)
    parser.add_argument('--phase_knn',         action='store_true', default=False)
    parser.add_argument('--adaptive_k',        action='store_true', default=False)
    parser.add_argument('--k_min',             type=int,   default=1)
    parser.add_argument('--multi_scale',       action='store_true', default=False)
    parser.add_argument('--low_freq_ratio',    type=float, default=0.25)
    parser.add_argument('--graph_depth',       type=int,   default=1)
    parser.add_argument('--dynamic_graph',     action='store_true', default=False)
    parser.add_argument('--phase_alpha_init',  type=float, default=0.5)
    parser.add_argument('--trend_graph',       action='store_true', default=False)
    parser.add_argument('--dropout',           type=float, default=0.0)

    # ---- training ----
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--decay_rate',    type=float, default=1e-4)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--grad_clip',     type=float, default=0.0)
    parser.add_argument('--patience',      type=int,   default=0)
    parser.add_argument('--optimizer',     type=str,   default='rmsprop',
                        choices=['rmsprop', 'adamw'])
    parser.add_argument('--scheduler',     type=str,   default='none',
                        choices=['none', 'plateau', 'cosine'])
    parser.add_argument('--min_lr',        type=float, default=1e-6)

    # ---- misc ----
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',   type=int, default=42)

    return parser.parse_args(args_list) if args_list is not None else parser.parse_args()