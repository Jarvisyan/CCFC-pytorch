import copy
import torch
import argparse
from tqdm import tqdm
from os import path, makedirs

from model import setup_seed, get_dataloaders, pretrain, asymmetric_loss, symmetric_loss, get_global_centroids, clustering_by_cosine_similarity



if __name__ == "__main__":
    parser = argparse.ArgumentParser('CCFC')
    parser.add_argument('--data_root', default= '../data',type=str, help='path to dataset directory')
    parser.add_argument('--exp_dir', default='./outputs', type=str, help='path to experiment directory')
    
    parser.add_argument('--trial', type=str, default='v0', help='trial id')
    parser.add_argument('--seed', type=int, default = 66, help='random seed')
    
    parser.add_argument('--proj_hidden_dim', default = 256, type=int, help='feature dimension')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
    parser.add_argument('--latent_dim', default = 256, type=int, help='feature dimension')
    parser.add_argument('--pre_hidden_dim', default = 16, type=int, help='feature dimension')
    
    parser.add_argument('--k', type = int, default = 10, help='the number of clusters')
    parser.add_argument('--lbd', type=float, default = 0.001, help='trade-off hyper')
    parser.add_argument('--p', type=float, default = 0., help='non-iid level')
    parser.add_argument('--lr', type=float, default = 0.001, help='learning rate')
    
    parser.add_argument('--batch_size', type=int, default = 128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default = 3, help='num of workers to use')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    
    if not path.exists(args.data_root):
        makedirs(args.data_root)    
    
    trial_dir = path.join(args.exp_dir, args.trial)
    if not path.exists(trial_dir):
        makedirs(trial_dir)
    with open(path.join(trial_dir, "logs.txt"), 'a') as f:
        f.write(f'\n\n{vars(args)}\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Nets, test_loader, ground_truth_all, n = get_dataloaders(args)
    Nets, global_model, pseudo_labels = pretrain(Nets, args, test_loader, ground_truth_all, n, trial_dir, device)
    
    print(f'training on: {device}')
    for epoch in range(30):
        global_model.eval()
        global_w = global_model.state_dict()
        for i in range(args.k):
            # Download global model from (virtual) central server
            Nets[f'model_{i}'].load_state_dict(copy.deepcopy(global_w))
            Nets[f'model_{i}'].train()
            pbar = tqdm(Nets[f'train_loader_{i}'])
            for images, idxs in pbar:
                Nets[f'optim_{i}'].zero_grad()
                x = images.to(device)  
                z, p = Nets[f'model_{i}'](x)
                with torch.no_grad():
                    _, p_g = global_model(x)
                
                count = 0
                loss_cluster = 0
                labels = pseudo_labels[idxs]
                for j in torch.unique(labels):
                    idx_j = labels == j
                    if  sum(idx_j) > 1:
                        count += 1
                        loss_cluster += symmetric_loss(p[idx_j], z[idx_j])
                loss_cluster /= count
                
                loss_model = asymmetric_loss(p, p_g)
                loss = loss_cluster + args.lbd * loss_model
                
                loss.backward()
                Nets[f'optim_{i}'].step()
                
                if epoch == 0:
                    Nets[f'loss_ema_{i}'] = loss.item()
                else:
                    Nets[f'loss_ema_{i}'] = 0.95 * Nets[f'loss_ema_{i}'] + 0.05 * loss.item()
                _loss_ema = Nets[f'loss_ema_{i}']
                pbar.set_description(f"Epoch: {epoch} | client: {i} | loss_ema: {_loss_ema : .4f} | loss_cluster: {loss_cluster: .4f} | loss_model: {loss_model: .4f}")
        
        # Averaging the local models' parameters to get global model
        net_para = Nets[f'model_{0}'].state_dict()
        for key in net_para:
            global_w[key] = net_para[key] * len(Nets[f'd_{0}']) / n
        for i in range(1, args.k):
            net_para = Nets[f'model_{i}'].state_dict()
            for key in net_para:
                global_w[key] += net_para[key] * len(Nets[f'd_{i}']) / n 
        global_model.load_state_dict(copy.deepcopy(global_w)) #update global_model
        
        global_centroids = get_global_centroids(args, Nets, device)   #update global_centroids
        pseudo_labels, acc, nmi = clustering_by_cosine_similarity(test_loader, global_model, global_centroids, ground_truth_all.numpy(), device) 
        
        save_path = path.join(trial_dir, f"model_train_{int(args.p / 0.25)}.pt")
        torch.save(global_model.state_dict(), save_path)
        
        with open(path.join(trial_dir, "logs.txt"), 'a') as f:
            f.write(f'Seed: {args.seed} | Epoch: {epoch : 03d} | loss_ema: {_loss_ema : .4f} | loss_cluster: {loss_cluster: .4f} | loss_model: {loss_model: .4f} | ACC = {acc: .4f} | NMI = {nmi : .4f}\n')