import os
import torch
import random
import numpy as np
from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
import wandb
import datetime
from random import randint
number = str(randint(0, 10000))
now = datetime.datetime.now()

from src.nf.glow import Glow
from data_loader import ISIC


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
device = torch.device("cuda")
torch.cuda.empty_cache()
print("GPU used: ", torch.cuda.get_device_name(device))

def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("Seed: ", args.seed)
    fName = time.strftime("%Y%m%d_%H_%M")
    if not os.path.exists("saves/"):
        os.makedirs("saves/")

    if args.model == "glow":
        cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
        p = SourceFileLoader('cf', 'config_glow.py').load_module()
        model = Glow(p, shape=p.imShape, conditional=p.conditional).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)

    if args.data == "isic":
        dataset = ISIC(cf, benign=True, validation=False, test=False, gray=cf.grayscale)       
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    dir_save = args.model+args.data+'_'+number
    os.mkdir(f'saves/{dir_save}')


    wandb.init(
        # set the wandb entity where this run will be logged
        entity="l-abdi",
        # set the wandb project where this run will be logged
        project="Flow",

        # set wandb model name
        name=args.model+'_'+number+'_'+now.strftime("%Y%m%d"),
        
        config={

        "dataset": args.data,
        "model": args.model,
        "seed:": args.seed,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "resolution": cf.patch_size,
        "max_epochs": args.epochs,
        "max_patience:": args.patience,
        }
    )
    
    model = model.to(device)
    lowest = 1e7
    avg_grad = 0
    avg_likelihood = 0
    patience = 0
    model.train()
    ep = 1
  
    print('loader length', len(loader))
    print('dataset length',len(dataset))
    print(len(dataset)/args.batch_size)

    while True:
        if ep > args.epochs or patience > args.patience:  
            break
        ep_likelihood = []
        ep_grad = []
        for idx, (x, filename) in enumerate(loader):
            print(f"Epoch: {ep} Progress:      {round((idx * 100) / (len(loader)), 4)}% Loss: {lowest}, Likelihood:    {avg_likelihood} \n, Gradient:      {avg_grad}, Patience:      {patience}" , end="\r")
            wandb.log({"Epoch": ep, "Loss": lowest,"Likelihood": avg_likelihood, "Gradient": avg_grad, "Patience": patience})
            x = x.to(device)
            x.requires_grad_() # in order to derivate wrt x
            optimizer.zero_grad()

            res_dict = model(x, partial_level=args.level)
            likelihood = torch.mean(res_dict["likelihood"])
            likelihood.backward(retain_graph=True) 

            grad_loss = x.grad.clone() # gradient of log-likelihood wrt x
            grad_loss = grad_loss.flatten(start_dim=1) # flatten the gradient
            grad_loss = torch.linalg.norm(grad_loss, dim=1, ord=2) # l2 norm of the gradient
            
            optimizer.step()
            
            ep_grad.append(grad_loss.detach())
            ep_likelihood.append(likelihood.detach())
            avg_grad = torch.mean(torch.stack(ep_grad).requires_grad_()) # average the gradient over the batch
            avg_likelihood = torch.mean(torch.stack(ep_likelihood))

            avg_loss = avg_likelihood - args.alpha * avg_grad
            
            optimizer.zero_grad()
            avg_loss.backward(retain_graph=True)

        # check and update
        if lowest > avg_loss:    
            lowest = avg_loss
            path_model = f'saves/{dir_save}/best-{args.ratio}_{args.seed}_{fName}-{args.model}-{args.data}-{now.strftime("%Y%m%d_%H_%M")}.pt'
            torch.save(model.state_dict(), path_model)
            patience = 0
        else:
            patience += 1

        # save the model at the end of every epoch
        path_model = f'saves/{dir_save}/{ep}_{args.model}-{args.data}-{now.strftime("%Y%m%d_%H_%M")}.pt'
        torch.save(model.state_dict(), path_model)

        ep += 1


# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="glow", help='Flow model used')
parser.add_argument('--level', type=int, default=-1, help='Train level for some models')
parser.add_argument('--data', type=str, default="isic", help='Dataset used')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--alpha', type=float, default=2.0, help='Hyperparameter alpha; penalizes high scores, setting alpha to zero will result in vanilla likelihood optimalization')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

args = parser.parse_args()
main()