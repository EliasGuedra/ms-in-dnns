import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, KMNIST
import lightning as L
import matplotlib.pyplot as plt
import torch.distributions as D




def generate_gaussian_bimodal(means, stds, num_samples = 0, seed=43):
    torch.manual_seed(seed)
    n = len(means)
    mix = D.Categorical(torch.ones(n))  # Mixture weights (equal probability)
    comp = D.Independent(D.Normal(means, stds), 1)  # Gaussian Mixture Model
    z_dist = D.MixtureSameFamily(mix, comp)
    data = z_dist.sample((num_samples,))
    return data, z_dist, f"gaussian_{n}modal"
    

def generate_gaussian_bimodal_multivariate(means, covs, num_samples = 0, seed=43):
    torch.manual_seed(seed)
    n = len(means)
    mix = D.Categorical(torch.ones(n))  # Mixture weights (equal probability)
    comp = D.MultivariateNormal(loc=means, covariance_matrix=covs)
    z_dist = D.MixtureSameFamily(mix, comp)
    data = z_dist.sample((num_samples,))
    return data, z_dist, f"MVgaussian_{n}modal_"


def generate_gaussian(num_samples=1000, mean=(0, 0), std_dev=1.0, seed=42):
    torch.manual_seed(seed)
    data = std_dev*torch.randn(num_samples, 2) + torch.tensor(mean)
    return data

def generate_bimodal_uniform_rectangular(num_samples=1000, limits_0=((-3, -1), (-1, 1)), limits_1=((1, 3), (-1, 1)), seed=42):

    torch.manual_seed(seed)
    x_0 = torch.rand(num_samples // 2) * (limits_0[0][1] - limits_0[0][0]) + limits_0[0][0]
    y_0 = torch.rand(num_samples // 2) * (limits_0[1][1] - limits_0[1][0]) + limits_0[1][0]
    class_0 = torch.stack((x_0, y_0), dim=1)

    x_1 = torch.rand(num_samples // 2) * (limits_1[0][1] - limits_1[0][0]) + limits_1[0][0]
    y_1 = torch.rand(num_samples // 2) * (limits_1[1][1] - limits_1[1][0]) + limits_1[1][0]
    class_1 = torch.stack((x_1, y_1), dim=1)

    data = torch.cat([class_0, class_1], dim=0)
    return data



#Refactor such as generate_gaussian_bimodal
def generate_circular(num_samples=1000, center=(0, 0), radius = 1.0, std_dev=0.1, seed=42):
    
    torch.manual_seed(seed)
    angles = torch.rand(num_samples // 2) * 2 * torch.pi
    data = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1) + std_dev * torch.randn(num_samples // 2, 2) + torch.tensor(center)

    return data

def generate_circular_bimodal(num_samples=1000, center_0=(0, 0), center_1=(0, 0), radius_0=1.0, radius_1=2.0, std_dev=0.1, seed=42):
    torch.manual_seed(seed)
    n = 2
    mix = D.Categorical(torch.ones(n))  # Mixture weights (equal probability)
    ring_0 = RingDistribution(center_0, radius_0, std_dev)
    ring_1 = RingDistribution(center_1, radius_1, std_dev)
    rings = [ring_0, ring_1]
    samples_per_mode = num_samples // n
    data_modes = [rings[i].sample(samples_per_mode) for i in range(n)]
    data = torch.cat(data_modes, dim=0)

    class MixtureRingDistribution:
        def __init__(self, rings, mix):
            self.rings = rings
            self.mix = mix
            self.n = len(rings)
        def sample(self, sample_shape=torch.Size()):
            shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size([sample_shape])
            idx = self.mix.sample(shape)
            samples = []
            for i in range(self.n):
                num = (idx == i).sum().item()
                if num > 0:
                    samples.append(self.rings[i].sample(num))
            if samples:
                return torch.cat(samples, dim=0)
            else:
                return torch.empty(0, 2)
        def log_prob(self, value):
            log_probs = torch.stack([ring.log_prob(value) for ring in self.rings], dim=1)
            mix_probs = torch.log(self.mix.probs)
            return torch.logsumexp(log_probs + mix_probs, dim=1)

    z_dist = MixtureRingDistribution(rings, mix)
    return data, z_dist, f"circular_{n}modal"



class CustomDataModule(L.LightningDataModule):
    def __init__(self, data, batch_size=256, pred_batch_size=10, train_split=0.8):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.train_split = train_split

    def setup(self, stage, seed=42):
        self.dataset = TensorDataset(self.data)
        n_train = int(len(self.dataset) * self.train_split)
        n_val = len(self.dataset) - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        dummy_pred_dataset = TensorDataset(torch.zeros(self.pred_batch_size, 2))  # Dummy dataset for prediction
        return DataLoader(dummy_pred_dataset, batch_size=self.pred_batch_size)


class CustomDataModule(L.LightningDataModule):
    def __init__(self, data, batch_size=256, pred_batch_size=10, train_split=0.8,
                 train_idx=None, val_idx=None):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.train_split = train_split
        self.train_idx = train_idx
        self.val_idx = val_idx

    def setup(self, stage=None, seed=42):
        self.dataset = TensorDataset(self.data)

        if self.train_idx is not None and self.val_idx is not None:
            self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_idx)
            self.val_dataset = torch.utils.data.Subset(self.dataset, self.val_idx)
        else:
            n_train = int(len(self.dataset) * self.train_split)
            n_val = len(self.dataset) - n_train
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(seed)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        dummy_pred_dataset = TensorDataset(torch.zeros(self.pred_batch_size, 2))
        return DataLoader(dummy_pred_dataset, batch_size=self.pred_batch_size)
