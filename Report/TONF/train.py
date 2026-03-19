import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from datetime import datetime
import os
import sys
import pathlib as pl
import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint, Callback, EarlyStopping
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold



class FullReportCallback(Callback):
    def __init__(self, enabled, save_path, data, device, line=False):
        super().__init__()
        self.enabled = enabled
        self.save_path = save_path
        self.data = data
        self.device = device
        self.line = line

    def on_train_epoch_end(self, trainer, pl_module):
        if self.enabled:
            # Ensure the directory exists
            if self.save_path and not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            epoch = trainer.current_epoch
            file_name = f"FR_{epoch}"
            full_report_plot(pl_module, self.data, self.device, save_path=self.save_path, file_name=file_name, line=self.line)

import TONF
from TONF.data import CustomDataModule, generate_bimodal_uniform_rectangular, generate_gaussian_bimodal_multivariate
from TONF.model import NormalizingFlow2DModule, LearnablePriorNF2D, LearnablePriorNFND
from TONF.utils import get_wandb_key, args_to_flat_dict
from TONF.evaluation import full_report_plot
import torch
import torchvision.utils as vutils
import torch.distributions as D

from sklearn.model_selection import train_test_split

import pandas as pd

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def monte_carlo_entropy(x_samples, density_fn):
    p_vals = density_fn(x_samples)
    eps = 1e-12
    p_vals = torch.clip(p_vals, eps, None)
    H = -torch.mean(torch.log(p_vals))
    return H



def main(args):
    N = 28
    #%% CONFIG UPDATE
    seed = 0xDEADBEEF    
    print(f"Setting the random seed for reproducibility to {seed}")
    seed_everything(seed, workers=True)

    #Logging in to wandb and setting up the logger
    if "LOG_PATH" in os.environ:
        wandb_save_dir = os.path.dirname(os.environ["LOG_PATH"])
    else:
        wandb_save_dir = "."
    wandb.login(key=get_wandb_key())
    args.trainer.logger = WandbLogger(
        project="TONF", name=args.run_name, save_dir=wandb_save_dir
    )
    args.trainer.logger.experiment.config.update(args_to_flat_dict(args))



    #%% GENERATE DATA
    #means = (torch.rand(10, 2)*30) - 15
    #circular data
    #alphas = torch.linspace(0, 2*torch.pi, 61)
    #Does the concatenation with the zeroes make sense
    #ring = torch.stack((10*torch.sin(alphas), 10*torch.cos(alphas)), dim=1)
    #means = ring[:-1]

    
    #t = torch.linspace(0, 2*torch.pi, 500)
#
    #knot = torch.stack(
    #    (
    #        torch.sin(t) + 2*torch.sin(2*t),
    #        torch.cos(t) - 2*torch.cos(2*t),
    #        -torch.sin(3*t)
    #    ),
    #    dim=1
    #)
    
    #means = knot * 10
    #codimensions = torch.zeros(ring.shape[0], N-2)
    #print(f"Means shape: {means.shape}, codimensions shape: {codimensions.shape}")
    #means = torch.cat((knot, codimensions), dim=1)[:-1]
    #means = torch.cat((ring, codimensions), dim=1)[:-1]
    #means = (torch.rand(5, 2)*30) - 15

    #means = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    #means2 = torch.stack((torch.sin(alphas)*20, torch.cos(alphas)*5, torch.zeros(30)), dim=1)[:-1]
    #means = torch.cat((means, means2), dim=0)
    #stds = torch.ones((30, 2)) * 1.0
    #args.data.data, args.model.target_dist, distribution_name = TONF.data.generate_gaussian_bimodal(means=means, stds=stds, num_samples=1_000_000) 
    #covs = torch.eye(N).unsqueeze(0).repeat(means.shape[0], 1, 1) * 0.3

    #data = load_wine()
    #X = data.data
    #X = StandardScaler().fit_transform(X)
    #args.data.data = torch.tensor(X, dtype=torch.float32)
    #args.distribution_name = "wine"
    df = pd.read_csv("/Users/eliasguedra/Downloads/hepmass/all_train.csv", nrows=20_000)

    X = df.iloc[:, 1:].values

    # split FIRST
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # fit scaler only on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])

    # combine back
    X_scaled = np.zeros_like(X)
    X_scaled[train_idx] = X_train
    X_scaled[val_idx] = X_val

    args.data.data = torch.tensor(X_scaled, dtype=torch.float32)
    args.data.train_idx = train_idx
    args.data.val_idx = val_idx
    # load data
    #df = pd.read_csv("/Users/eliasguedra/Downloads/hepmass/all_train.csv", nrows=20_000)
    ## drop label column (first column)
    #X = df.iloc[:, 1:]
#
    ## normalize
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
#
    ## convert to torch tensor
    #args.data.data = torch.tensor(X_scaled, dtype=torch.float32)
    args.distribution_name = "hepmass"

    #args.data.data, args.model.target_dist, distribution_name = TONF.data.generate_gaussian_bimodal_multivariate(means=means, covs=covs, num_samples=100_000)
    args.LRmodel.target_dist = args.model.target_dist
    dm      = CustomDataModule(**vars(args.data))
    #entropy = monte_carlo_entropy(args.data.data[:10000], lambda x: torch.exp(args.model.target_dist.log_prob(x)))
    #print(f"Estimated entropy of the target distribution: {entropy.item():.4f}")
    #%% NormalizingFlow2DModule setup
    means_z_dist = torch.tensor([[0.0,0.0]])
    stds_z_dist = torch.tensor([[1.0, 1.0]])
    _, args.model.z_dist, z_distribution_name = TONF.data.generate_gaussian_bimodal(means=means_z_dist, stds=stds_z_dist, num_samples=10)   
    model = NormalizingFlow2DModule(**vars(args.model))

    #%% LearnablePriorNF2D setup

    #Latent prior
    #means_z_dist = (torch.rand(5, N) * 30) - 15
    # Try choose the means randomly from the target data, this way we ensure that the components are well placed in the latent space.
    n_latent_components = 10
    train_idx = torch.tensor(args.data.train_idx)
    sampled = torch.randperm(len(train_idx))[:n_latent_components] # AVOID DATALEAKAGE!
    indices = train_idx[sampled]
    #indices = torch.randperm(args.data.data.shape[0])[:n_latent_components]
    means_z_dist = args.data.data[indices]
    #zero_mean = torch.tensor([[0.0]*N], device=means_z_dist.device)
    #means_z_dist = torch.cat([means_z_dist, zero_mean], dim=0)
    #means_z_dist = (torch.rand(n_latent_components, N)*2) - 15
    # offset so that mean is (15, 15) and not all in the same quadrant
    #means_z_dist -= means_z_dist.mean(dim=0)
    #means_z_dist = torch.tensor([[0.0]*N])
    stds_z_dist  = torch.tensor([[1.0]*N]*means_z_dist.shape[0])
    _, _, z_distribution_name = TONF.data.generate_gaussian_bimodal(means=means_z_dist, stds=stds_z_dist, num_samples=10)   

    args.LRmodel.means = means_z_dist
    args.LRmodel.stds  = stds_z_dist
    args.LRmodel.dim = N
    #args.LRmodel.target_entropy = entropy
    LRmodel = LearnablePriorNFND(**vars(args.LRmodel))


    #%%

    run_data = f"{args.model_type}_DIM={N}_{None}_lr={args.model.lr}-{None}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.extra_info}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.trainer.callbacks = [
        RichModelSummary(max_depth=2),
        RichProgressBar(),
        ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"//Users/eliasguedra/Documents/GitHub/ms-in-dnns/Report/TONF/Interesting runs/{run_data}",
            mode="min",
            save_last=True,
            filename="epoch={epoch}-val_loss={val/loss:.4f}-standard_data-nobridge",
            auto_insert_metric_name=False,
        ),
        FullReportCallback(
            enabled=getattr(args, "full_report", False),
            save_path=f"/Users/eliasguedra/Documents/GitHub/ms-in-dnns/Report/TONF/Interesting runs/{run_data}",
            data=args.data.data[:5000],
            device=device,
            line=False,
        ),
        EarlyStopping(
        monitor="val/loss",
        patience=20,
        mode="min",
        verbose=True
        ),
    ]

    trainer = Trainer(**vars(args.trainer))
    print("Starting training...")
    print(f"Training for a maximum of {args.trainer.max_epochs} epochs with accelerator {args.trainer.accelerator}...")
    if args.model_type == "LearnablePriorNF2D":
        print("--" * 50)
        print("\033[92mUsing LearnablePriorNF2D model.\033[0m")
        print("--" * 50)
        trainer.fit(LRmodel, datamodule=dm)
    else:
        print("--" * 50)
        print("\033[92mUsing NormalizingFlow2DModule model.\033[0m")
        print("--" * 50)
        trainer.fit(model, datamodule=dm)
#
    #X = args.data.data
    #kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    #fold_val_losses = []
    #for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#
    #    print(f"===== Fold {fold+1} =====")
#
    #    dm = CustomDataModule(
    #        data=X,
    #        batch_size=args.data.batch_size,
    #        pred_batch_size=args.data.pred_batch_size,
    #        train_idx=train_idx,
    #        val_idx=val_idx
    #    )
#
    #    model = LearnablePriorNFND(**vars(args.LRmodel))
#
    #    trainer = Trainer(**vars(args.trainer))
    #    trainer.fit(model, datamodule=dm)
#
    #    val_loss = trainer.callback_metrics["val/loss"].item()
#
    #    print(f"Fold {fold+1} val_loss: {val_loss:.4f}")
#
    #    fold_val_losses.append(val_loss)
#
    #avg_loss = np.mean(fold_val_losses)
    #std_loss = np.std(fold_val_losses)
#
    #print("\n===== Cross-Validation Result =====")
    #print(f"Average val_loss: {avg_loss:.4f}")
    #print(f"Std val_loss: {std_loss:.4f}")








if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(Trainer, "trainer")
    parser.set_defaults({"trainer.max_epochs": 10, "trainer.num_sanity_val_steps": 2, "trainer.accelerator": "cpu"})

    parser.add_lightning_class_args(LearnablePriorNFND, "LRmodel")
    parser.add_lightning_class_args(NormalizingFlow2DModule, "model")

    parser.add_lightning_class_args(CustomDataModule, "data")

    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    #parser.add_argument("--ckpt_path", type=str, default="Report/TONF/RUNS/gaussian_3modal_gaussian_3modal_0.001_3blocks_seed3735928558_20260304_135633/epoch=9-val_loss=4.47-standard_data-nobridge.ckpt")

    parser.add_argument("--extra_info", type=str, default="")

    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--test",  type=bool, default=False)
    parser.add_argument("--interpolation",  type=bool, default=True)
    parser.add_argument("--full_report", type=bool, help="Run full_report after every epoch.", default=False)
    parser.add_argument("--model_type", type=str, default="NormalizingFlow2DModule", choices=["LearnablePriorNF2D", "NormalizingFlow2DModule"], help="Select model type.")
    check_model_parser = parser.parse_args()

    args = parser.parse_args()
    main(args)

