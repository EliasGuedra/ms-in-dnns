

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import wandb
import matplotlib.pyplot as plt



class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CouplingLayer(nn.Module):
    def __init__(self, network, mask):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer("mask", mask)

    def forward(self, z, ldj, reverse=False):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
        """
        # Apply network to masked input
        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = nn_out.chunk(2, dim=1)


        # Mask outputs (only transform the second part)
        t = t * (1 - self.mask)
        s = s * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1])

        return z, ldj

class NormalizingFlow2D(nn.Module):
    def __init__(self):
        super().__init__()
        layer_list = []
        
        layer_list += [
            CouplingLayer(NN(), mask=torch.tensor([0, 1])),
            CouplingLayer(NN(), mask=torch.tensor([1, 0])),
        ]
        self.layers = nn.ModuleList(layer_list)


    def forward(self, inputs, reverse=False):
        
        if isinstance(inputs, (list, tuple)):
            z = inputs[0]
        else:
            z = inputs

        ldj = torch.zeros(z.shape[0], device=z.device)

        for layer in self.layers if not reverse else reversed(self.layers):
            z, ldj = layer(z, ldj, reverse=reverse)
        
        return z, ldj
    
class NormalizingFlow2DModule(L.LightningModule):
    def __init__(self, 
                 lr=1e-3, 
                 res_blocks=3, 
                 z_dist=None, 
                target_dist=None, 
                target_entropy=None):
        
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.target_dist = target_dist

        if z_dist is None:
            mean = torch.zeros(2)
            cov = torch.tensor([[1.0, 0], [0, 1.0]])
            self.z_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        else:
            self.z_dist = z_dist
        
        self.model = NormalizingFlow2D()
        self.target_entropy = target_entropy
        
    def get_latent_distribution(self):
        return self.z_dist
    
    def forward(self, x, reverse):
        return self.model(x, reverse=reverse)

    def nll(self, z, ldj, mean=True):
        # the z_dist is one-dimensional and applied element-wise
        # the joint is the product (elements are independent), so sum the logs
        z_dist = self.get_latent_distribution()
        log_qz = z_dist.log_prob(z)
        log_qx = ldj + log_qz
        nll = -log_qx
        nll = nll.mean() if mean else nll
        return nll

    def log_samples(self, n_samples, stage="pred"):
        z = self.z_dist.sample(sample_shape=(n_samples, 2)).to(self.device)
        x, ldj = self(z, reverse=True)
        return x

    def shared_step(self, batch, metric_prefix):
        x = batch
        z, ldj = self(x, reverse=False)
        loss = self.nll(z, ldj)
        self.log(f"{metric_prefix}/loss", loss, on_epoch=True, on_step=False)
        
        if self.target_entropy is not None:
            self.log(f"{metric_prefix}/Normalized_loss", self.target_entropy/loss, on_epoch=True, on_step=False)

        if metric_prefix in ["train", "val"]:
            self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)
        
        #entropy = torch.distributions.Categorical(
        #    logits=self.mixture_logits
        #).entropy()
#
        #self.log(f"entropy", entropy, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "best")

    def on_fit_start(self):
        self.val_samples_table = wandb.Table(columns=["image", "epoch"])

    def on_fit_end(self):
        wandb.log({"val/samples": self.val_samples_table})

    def predict_step(self, batch, batch_idx):
        #plot_model(self.model, batch[0], self.device)
        return self.log_samples(batch[0].shape[0], "pred")

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
    
class LearnablePriorNF2D(NormalizingFlow2DModule):
    def __init__(self, means, stds, lr=1e-3, res_blocks=3, target_dist=None, lam=1, lamdecay=0.0001, cov = None, target_entropy=None, trainable_prior=True):


        super().__init__(
            lr=lr,
            res_blocks=res_blocks,
            z_dist=None,
            target_dist=target_dist,
            target_entropy=target_entropy
        )
        
        if cov is None:
            cov = torch.eye(means.shape[1])

        self.K = means.shape[0]

        # Fixed Gaussian component parameters
        self.lam = lam
        self.lamdecay = lamdecay
        if trainable_prior:
            self.mixture_logits = nn.Parameter(torch.zeros(self.K))
            self.means = nn.Parameter(means)
            L = torch.linalg.cholesky(cov)
            self.L_unconstrained = nn.Parameter(L.repeat(self.K, 1, 1))
        else:
            self.register_buffer("means", means)
            self.register_buffer("stds", stds)
            self.register_buffer("mixture_logits", torch.zeros(self.K))
            self.register_buffer("L_unconstrained", torch.linalg.cholesky(cov).repeat(self.K, 1, 1))

        #self.stds = nn.Parameter(stds)
        

        #self.register_buffer("means", means)
        #self.register_buffer("stds", stds)

        self.max_entropy = torch.log(torch.tensor(self.K, device=self.mixture_logits.device, dtype=torch.float32))

    def nll(self, z, ldj, mean=True):
        nll = super().nll(z, ldj, mean)

        if self.K == 1:
            return nll

        entropy = torch.distributions.Categorical(
            logits=self.mixture_logits
        ).entropy()
        
        decay = torch.exp(torch.tensor(-self.lamdecay * self.global_step))
        decay = self.lam * decay / self.max_entropy
        loss = nll - entropy*decay
        #print(f"nll: {nll.item():.4f}, entropy: {entropy.item():.4f}, decay: {decay.item():.4f}, loss: {loss.item():.4f}")
        return loss

    def get_L(self):
        L = torch.tril(self.L_unconstrained)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag) + 1e-4
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return L
    
    def get_latent_distribution(self):
        mix = torch.distributions.Categorical(logits=self.mixture_logits)

        comp = torch.distributions.MultivariateNormal(
            loc=self.means,        # (K,2)
            scale_tril=self.get_L()      # (K,2,2)
        )
        return torch.distributions.MixtureSameFamily(mix, comp)


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
class NormalizingFlowND(NormalizingFlow2D):
    def __init__(self, dim = 2):
        super().__init__()
        layer_list = []
        #Have alternating mask tensors of size dim
        mask1 = torch.tensor([0] * (dim // 2) + [1] * (dim - dim // 2))
        mask2 = 1 - mask1
        layer_list += [
            CouplingLayer(NN(input_dim=dim, output_dim=dim*2), mask=mask1),
            CouplingLayer(NN(input_dim=dim, output_dim=dim*2), mask=mask2),
        ]
        self.layers = nn.ModuleList(layer_list)

class LearnablePriorNFND(LearnablePriorNF2D):
    def __init__(self, means, stds, dim = 2, lr=1e-3, res_blocks=3, target_dist=None, lam=1, lamdecay=0.0005, cov = None, target_entropy=None, trainable_prior=True):
        if cov is None:
            cov = torch.eye(means.shape[1])
        super().__init__(means, stds, lr, res_blocks, target_dist, lam, lamdecay, cov, target_entropy=target_entropy, trainable_prior=trainable_prior)
        self.model = NormalizingFlowND(dim = dim)