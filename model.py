import copy
import torch
import random
import argparse
import numpy as np
from os import path
from torch import nn
from tqdm import tqdm
from PIL import ImageFilter
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler

from sklearn.cluster import KMeans
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def construct_workers(Nets, M, p, ground_truth):
    #M: number of clients/workers
    setup_seed(20)
    if p == 1:
        for i in range(M):
            Nets[f'd_{i}'] = np.where(ground_truth == i)[0]
            print(torch.bincount(ground_truth[Nets[f'd_{i}']]), len(ground_truth[Nets[f'd_{i}']]))
    else:
        n = int(ground_truth.shape[0] / M) #the number of samples in each client
        idx_rest = np.zeros(0, int)
        for i in range(M): #assign n*p samples from the single cluster to each client separately
            Nets[f'd_{i}'] = np.where(ground_truth == i)[0][ : round(n * p)]
            d_i_rest = np.where(ground_truth == i)[0][round(n * p) : ]
            idx_rest = np.concatenate((idx_rest, d_i_rest))
        shuffle_idx = torch.randperm(idx_rest.shape[0]) #assign n*(1-p) samples from several clusters to each client evenly
        idx_rest_shuffled = idx_rest[shuffle_idx]
        assert round(n * p) + round(n * (1-p)) == n
        idx1, idx2 = 0, round(n * (1-p))
        for i in range(M):
            Nets[f'd_{i}'] = np.concatenate((Nets[f'd_{i}'], idx_rest_shuffled[idx1 : idx2]))
            idx1 = idx2
            idx2 += round(n * (1-p))
            assert np.unique(Nets[f'd_{i}']).shape[0] == n
            print(torch.bincount(ground_truth[Nets[f'd_{i}']]), len(ground_truth[Nets[f'd_{i}']]))

def get_dataloaders(args):
    train_dataset = MNIST(root = args.data_root, train = True, download = True, transform = transforms.ToTensor())
    test_dataset = MNIST(root = args.data_root, train = False, download = True, transform = transforms.ToTensor())
    ground_truth_all = torch.cat((train_dataset.targets, test_dataset.targets))
    print(f'label distribution: {torch.bincount(ground_truth_all)}')
    
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    n = n_train + n_test
    train_dataset.targets = torch.arange(0, n_train)#index of images and labels  pseudo_labels[ : n_train]
    test_dataset.targets = torch.arange(n_train, n)#                             pseudo_labels[n_train : ]    
    combined_dataset = ConcatDataset([train_dataset, test_dataset]) 
    
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.5, 1.)), 
        transforms.RandomRotation(10), 
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor()
    ])
    train_dataset_aug = MNIST(root = args.data_root, train = True, download = True, transform = TwoCropsTransform(train_transforms))
    test_dataset_aug = MNIST(root = args.data_root, train = False, download = True, transform = TwoCropsTransform(train_transforms))
    combined_dataset_aug = ConcatDataset([train_dataset_aug, test_dataset_aug])
    
    Nets = locals()
    construct_workers(Nets, M = args.k, p = args.p, ground_truth = ground_truth_all)
    for i in range(args.k):
        sampler_i = SubsetRandomSampler(Nets[f'd_{i}'])
        Nets[f'pretrain_loader_{i}'] = DataLoader(combined_dataset_aug, args.batch_size, sampler = sampler_i,
                              num_workers = args.num_workers, pin_memory = True)
        Nets[f'train_loader_{i}'] = DataLoader(combined_dataset, args.batch_size, sampler = sampler_i,
                              num_workers = args.num_workers, pin_memory = True)
    test_loader = DataLoader(combined_dataset, args.batch_size, shuffle = False, num_workers = args.num_workers)
    
    return Nets, test_loader, ground_truth_all, n



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        out_dim = in_dim

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class SimSiam(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        #conv_block: 1*28*28 -> 64*14*14 -> 128*7*7 -> 256*3*3
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.projector = projection_MLP(256 * 3 * 3, args.proj_hidden_dim, args.latent_dim, args.num_proj_layers)
        self.predictor = prediction_MLP(args.latent_dim, args.pre_hidden_dim)
           
    def forward(self, x):
        z = self.projector(self.backbone(x).view(-1, 256 * 3 * 3))
        p = self.predictor(z)    
        
        return z, p
    
def asymmetric_loss(p, z): #sample-level
    z = z.detach()  # stop gradient
    return - F.cosine_similarity(p, z, dim=-1).mean()

def symmetric_loss(p, z): #cluster-level
    z = z.detach()  # stop gradient
    #ipdb.set_trace()
    z_norm, p_norm = F.normalize(z), F.normalize(p)
    return - torch.mm(z_norm, p_norm.T).mean()

def get_centroids(latent_z, nClusters):
    kmeans = KMeans(n_clusters = nClusters).fit(latent_z)
    return kmeans.cluster_centers_
    
def get_global_centroids(args, Nets, device):
    local_latent_z_ls = [] 
    local_centroids_ls = []
    for i in range(args.k):
        Nets[f'model_{i}'].eval()
        
        latent_z = []
        with torch.no_grad():
            for x, _ in Nets[f'train_loader_{i}']:
                x = x.to(device)
                z = F.normalize(Nets[f'model_{i}'](x)[0]) #check here
                latent_z.append(z)
                    
        latent_z = torch.cat(latent_z).cpu().numpy()
        local_latent_z_ls.append(latent_z)
        
        local_centroids = get_centroids(latent_z, args.k)
        local_centroids_ls.append(local_centroids)
       
    local_centroids_all = np.concatenate(local_centroids_ls) ##check here 
    global_centroids = get_centroids(local_centroids_all, args.k)
    return global_centroids

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w) 
    return sum([w[i,j] for i,j in zip(ind[0], ind[1])])*1.0/Y_pred.size, w

def clustering_by_cosine_similarity(test_dataloader, global_model, global_centroids, ground_truth, device):
    global_model.eval()
    latent_z = []
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = x.to(device)
            z = F.normalize(global_model(x)[0])
            latent_z.append(z)
    latent_z_all = torch.cat(latent_z, 0).cpu().numpy()
    pred = cosine_similarity(latent_z_all, global_centroids).argmax(1)
    acc, nmi = cluster_acc(pred, ground_truth)[0], NMI(ground_truth, pred)
    print(f'acc: {acc: .4f} | nmi: {nmi: .4f}')
    return torch.from_numpy(pred), acc, nmi


def pretrain(Nets, args, test_loader, ground_truth_all, n, trial_dir, device):
    global_model = SimSiam(args).to(device)
    save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}.pt")
    if not path.exists(save_path):
        for i in range(args.k):
            Nets[f'model_{i}'] = SimSiam(args).to(device)
            Nets[f'optim_{i}'] = torch.optim.Adam(Nets[f'model_{i}'].parameters(), lr = args.lr)
    
        print(f'pretraining on: {device}')
        loss_ema = None
        for epoch in range(100):
            global_model.eval()
            global_w = global_model.state_dict() 
            for i in range(args.k):
                if epoch > 0:
                    Nets[f'model_{i}'].load_state_dict(copy.deepcopy(global_w)) ##check here
                Nets[f'model_{i}'].train()
                pbar = tqdm(Nets[f'pretrain_loader_{i}'])
                for images, _ in pbar:
                    Nets[f'optim_{i}'].zero_grad()
                    x1 = images[0].to(device)
                    x2 = images[1].to(device)  
                    z1, p1 = Nets[f'model_{i}'](x1)
                    z2, p2 = Nets[f'model_{i}'](x2)
                    
                    with torch.no_grad():
                        _, p1_g = global_model(x1)
                        _, p2_g = global_model(x2)
                    
                    loss_sample = 0.5 * asymmetric_loss(p1, z2) + 0.5 * asymmetric_loss(p2, z1)
                    loss_model = 0.5 * asymmetric_loss(p1, p1_g) + 0.5 * asymmetric_loss(p2, p2_g)
                    loss = loss_sample + args.lbd * loss_model
                    loss.backward()
                    Nets[f'optim_{i}'].step()
                    
                    if loss_ema is None:
                        loss_ema = loss.item()
                    else:
                        loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                    pbar.set_description(f"Epoch: {epoch} | client: {i} | loss_ema: {loss_ema : .4f} | loss_sample: {loss_sample: .4f} | loss_model: {loss_model: .4f}")
            
            # Averaging the local models' parameters to get global model
            net_para = Nets[f'model_{0}'].state_dict()
            for key in net_para:
                global_w[key] = net_para[key] * len(Nets[f'd_{0}']) / n
            
            for i in range(1, args.k):
                net_para = Nets[f'model_{i}'].state_dict()
                for key in net_para:
                    global_w[key] += net_para[key] * len(Nets[f'd_{i}']) / n
            global_model.load_state_dict(copy.deepcopy(global_w))
            
            global_centroids = get_global_centroids(args, Nets, device)
            pseudo_labels, acc, nmi = clustering_by_cosine_similarity(test_loader, global_model, global_centroids, ground_truth_all.numpy(), device) 
            
            #save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}.pt")
            torch.save(global_model.state_dict(), save_path)
            
            with open(path.join(trial_dir, "logs.txt"), 'a') as f:
                f.write(f'Seed: {args.seed} | Epoch: {epoch : 03d} | loss_ema: {loss_ema : .4f} | loss_sample: {loss_sample: .4f} | loss_model: {loss_model: .4f}\n')    
    
    checkpoint = torch.load(save_path, map_location=device)
    global_model.load_state_dict(checkpoint)
    for i in range(args.k):
        Nets[f'model_{i}'] = copy.deepcopy(global_model)
        Nets[f'optim_{i}'] = torch.optim.Adam(Nets[f'model_{i}'].parameters(), lr = args.lr) 
        
    global_centroids = get_global_centroids(args, Nets, device)
    pseudo_labels, acc, nmi = clustering_by_cosine_similarity(test_loader, global_model, global_centroids, ground_truth_all.numpy(), device)    
    return Nets, global_model, pseudo_labels
    
    