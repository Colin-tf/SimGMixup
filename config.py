import argparse
import torch
  

def config():
    parser = argparse.ArgumentParser(description='Mixup')
    parser.add_argument('--device', type=int, default=3, help='-1 for cpu, 0 for first gpu, 1 for second gpu, etc.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')
    parser.add_argument('--log_steps', type=int, default=1, help='number of steps to log')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--eval_steps', type=int, default=5, help='number of steps to evaluate')
    parser.add_argument('--exp_tag', type=str, default="", help='Experiment Tag')
    parser.add_argument('--dataset', type=str, default="computers", help='Dataset Name')
    parser.add_argument('--num_per_class', type=int, default=-1, help='Number of samples per class')
    parser.add_argument('--early_stop', type=int, default=100, help='Early Stop')
    parser.add_argument('--task', type=str, default="node", help='Task Name [node, graph]')
    
    # for Backbone GNN
    parser.add_argument('--model', type=str, default='GCN', help='Name of the GNN model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--hidden_channels', type=int, default=128, help='number of hidden channels')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay rate')
    
    # for mixup modules configuration
    parser.add_argument('--mixup_mode', type=str, default="input", help='Mixup Method [input, hidden]')
    parser.add_argument('--mixup', type=str, default="mixup", help='Mixup Method [agmixup, none]')
    parser.add_argument('--edge_mixer', type=str, default="sum", help='Edge Mixer [sum, sample, biased]')
    parser.add_argument('--feat_mixer', type=str, default="NN", help='Feature Mixer [vanilla, saliency, global]')
    parser.add_argument('--mixing_ratio', type=str, default="random", help='Mixing Ratio [context, random, static]')
    parser.add_argument('--sampler', type=str, default="random", help='Sampler [random, static]')
    parser.add_argument('--pairing_mode', type=str, default="LL", help='Pairing Mode [LL, LU, UU]')
    
    # parameters for mixup methods
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shrink_ratio', type=float, default=1)
    parser.add_argument('--beta_alpha', type=float, default=2)
    parser.add_argument('--init_lam', type=float, default=0.5)
    parser.add_argument('--hop', type=int, default=2, help="")
    
    return parser.parse_args()

args = config()
if args.device >= 0:
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
device = torch.device(device)

    
if __name__ == "__main__":
    import run_node_task
    run_node_task.main()