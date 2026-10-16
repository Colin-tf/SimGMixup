import torch
from models import *
from mixup_modules.graph_mixers import *
from mixup_modules.extractor import *
from mixup_modules.edge_mixers import *
from mixup_modules.feat_mixers import *
from mixup_modules.samplers import *
from mixup_modules.mixing_ratios import *
from mixup_modules.mixup_loss import mixup_loss_func
from graph import get_dataset_nodes_task
from config import args, device
from utils import seed_setting
from logger import Logger
from torch.optim.lr_scheduler import StepLR

MIEXER_CONFIG_DICT = {
    "edge_mixer" : {
        "sum": SumEdgeMixer,
        "sample": SampleEdgeMixer,
        "biased": BiasedEdgeMixer
    },
    "feat_mixer" : {
        "NN": NNFeatMixer,
        "NG": NGFeatMixer,
        "GG": GGFeatMixer,
        "GN": GNFeatMixer,
    },
    "sampler" : {
        "random": RandomSampler,
        "static": StaticSampler
    },
    "mixing_ratio" : {
        "static": StaticMixingRatio,
        "random": RandomMixingRatio,
    }
}

GNN_CONFIG_DICT = {
    "GCN": GCN,
    "GAT": GAT,
    "SAGE": SAGE,
    "GIN": GIN
} 

    
def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_pred = out.argmax(1)
        train_acc = y_pred[data.train_mask].eq(data.y[data.train_mask]).float().mean().item()
        val_acc = y_pred[data.val_mask].eq(data.y[data.val_mask]).float().mean().item()
        test_acc = y_pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()
    
    return train_acc, val_acc, test_acc
        
        
def train(model, data, optimizer, mixers=None, is_sparse:bool=False):
    
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    data.x.requires_grad_(True)
    # Zero gradients for optimizer
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)

    # Compute cross-entropy loss
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    classification_loss = loss.item()
    loss.backward()
    optimizer.step()
    
    mixup_loss_all = 0
    input_grad = data.x.grad.clone()
    if len(mixers) > 0:
        for mixer in mixers:
            mixed_graph_loader = mixer(
                                    g=data, 
                                    p=out.softmax(-1).detach(), 
                                    h=model.get_embedding().detach(),
                                    grad=input_grad,
                                    sample_ratio=args.shrink_ratio,
                                    task="node"
                                    )
            if len(mixed_graph_loader) > 0:
                for mixed_graph in mixed_graph_loader:
                    optimizer.zero_grad()
                    mixed_graph = mixed_graph.to(device)
                    mixed_out = model.mixup_forward(mixed_graph, is_sparse=is_sparse, mixup_mode=mixer.mixup_mode)
                    
                    if mixer.mixup_mode == "input":
                        # for input mixer, we need to re-assign the output based on the mapping of central nodes
                        mixed_out = mixed_out[mixed_graph.mapping + mixed_graph.ptr[:-1]]
                        
                    mixup_loss = 0.5*mixup_loss_func(logits=mixed_out,
                                                    y_l=mixed_graph.y_l,
                                                    y_r=mixed_graph.y_r,
                                                    lam=mixed_graph.lam)
                    mixup_loss.backward()
                    optimizer.step()
                    mixup_loss_all += mixup_loss.item()
                mixup_loss_all /= len(mixed_graph_loader)
        mixup_loss_all /= len(mixers)

    train_loss = classification_loss + mixup_loss_all
    return train_loss

def main(mixup_conf: Dict = None):
    logger = Logger(args)
    
    for run in range(args.runs):
        seed_setting(args, run+1)
        
        data, is_sparse = get_dataset_nodes_task(args.dataset, args.num_per_class)
        
        # initialize model
        model = GNN_CONFIG_DICT[args.model](
            in_channels=data.num_features, 
            hid_channels=args.hidden_channels, 
            out_channels=data.num_classes, 
            dropout=args.dropout,
            num_layers=args.num_layers, 
            task=args.task).to(device)
        model.reset_parameters()
        model.build_mixup_forward_layers(mode=args.mixup_mode.lower())
        
        # initialize mixers
        mixers = []
        if args.mixup.lower() == "none":
            mixer = None
        else:
            mixup_mode_space = args.mixup_mode.split(",")
            edge_mixer_space = args.edge_mixer.split(",")
            feat_mixer_space = args.feat_mixer.split(",")
            sampler_space = args.sampler.split(",")
            mixing_ratio_space = args.mixing_ratio.split(",")
            pairing_mode_space = args.pairing_mode.split(",")
            for mixup_mode in mixup_mode_space:
                for edge_mixer in edge_mixer_space:
                    for feat_mixer in feat_mixer_space:
                        for sampler in sampler_space:
                            for mixing_ratio in mixing_ratio_space:
                                for pairing_mode in pairing_mode_space:
                                    if mixup_mode == "input":
                                        mixer=InputMixer(
                                            edge_mixer=MIEXER_CONFIG_DICT["edge_mixer"][edge_mixer.strip()](), 
                                            feat_mixer=MIEXER_CONFIG_DICT["feat_mixer"][feat_mixer.strip()](), 
                                            sampler=MIEXER_CONFIG_DICT["sampler"][sampler](mode=pairing_mode.strip()), 
                                            mixing_ratio=MIEXER_CONFIG_DICT["mixing_ratio"][mixing_ratio.strip()](pair_wise=False, init_lam=args.init_lam, beta_alpha=args.beta_alpha), 
                                            extractor=Extractor(to_folder=f"./mixed_graphs/{data.name}", hop=args.hop),
                                            batch_size=args.batch_size
                                        )
    
                                    mixers.append(mixer)
        
        # initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        
        best_test_acc = 0
        best_train_acc = 0
        best_val_acc = 0
        es = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, optimizer, mixers=mixers, is_sparse=is_sparse)

            # if epoch % args.eval_steps == 0:
            train_acc, val_acc, test_acc = test(model, data)
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * val_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')
            
            scheduler.step()
            # select the result based on valdation set
            if val_acc > best_val_acc:
                es = 0
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_train_acc = train_acc
            else:
                es += 1
                if es >= args.early_stop:
                    break
                
        logger.add_result(run, best_test_acc*100, "test")
        logger.add_result(run, best_val_acc*100, "val")
        logger.add_result(run, best_train_acc*100, "train")
        
    logger.print_statistics()
    
if __name__ == "__main__":
    print(args)
    main()