
import torch
import matplotlib.pyplot as plt
import TONF
from TONF.data import CustomDataModule, generate_gaussian, generate_gaussian_bimodal
from TONF.model import LearnablePriorNFND, NormalizingFlow2DModule, LearnablePriorNF2D
import os 

def plot_model(model, data, device, reverse=False):
    model.eval()

    start = torch.tensor((0, 0), device=device)
    end = torch.tensor((5, 5), device=device)

    t = torch.linspace(0, 1, steps=100, device=device).unsqueeze(1)
    bridge = start + t * (end - start)  

    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).to(device)
        z, _ = model(x, reverse=reverse)
        z = z.cpu().numpy()

        z_bridge, _ = model(bridge, reverse=reverse)
        z_bridge = z_bridge.cpu().numpy()


    plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.5, s=1)
    plt.plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=2, label='Bridge', color='red')
    plt.title("Transformed samples in latent space")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.grid()
    plt.axis("equal")
    plt.show()

def plot_ljd_heatmap(model, device):
    # Compute the log-determinant of the Jacobian for heatmap
    model.eval()
    with torch.no_grad():
        grid = torch.meshgrid(torch.linspace(-8, 8, 1000), torch.linspace(-8, 8, 1000))
        x = torch.stack(grid, dim=-1).reshape(-1, 2).to(device)

        _, ldj = model(x, reverse=False)
        ldj = ldj.cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(ldj.reshape(1000, 1000), extent=(-8, 8, -8, 8), origin='lower', cmap='viridis')
        plt.colorbar(label='Log-Determinant of Jacobian')
        plt.title("Log-Determinant of Jacobian Heatmap")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.grid()
        plt.show()

def plot_warping(model, device):
    # Plot the warping of a grid through the normalizing flow
    model.eval()

    # Create grid
    x = torch.linspace(-8, 8, 20)
    y = torch.linspace(-8, 8, 20)
    X, Y = torch.meshgrid(x, y)

    # Warp grid through the model
    with torch.no_grad():
        grid = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)
        warped_grid, _ = model(grid, reverse=False)
        warped_grid = warped_grid.cpu().numpy()
        Xw = warped_grid[:, 0].reshape(X.shape)
        Yw = warped_grid[:, 1].reshape(Y.shape)

    # Plot
    plt.figure(figsize=(10, 5))

    # Original grid
    plt.subplot(1, 2, 1)
    plt.title("Original Grid")
    plt.plot(X, Y, 'k', alpha=0.5)
    plt.plot(X.T, Y.T, 'k', alpha=0.5)
    plt.gca().set_aspect('equal')

    # Warped grid
    plt.subplot(1, 2, 2)
    plt.title("Warped Grid")
    plt.plot(Xw, Yw, 'k')
    plt.plot(Xw.T, Yw.T, 'k')
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.show()


def full_report_plot_3d(model, data, device, save_path=None, dpi=300, file_name="full_report", show=False):

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 6),
                            subplot_kw={'projection': '3d'})

    model.eval()
    with torch.no_grad():

        # Transform dataset
        x_data = torch.tensor(data, dtype=torch.float32).to(device)
        z_data, _ = model(x_data, reverse=False)

        x_np = x_data.cpu().numpy()
        z_np = z_data.cpu().numpy()

        # Bridge example
        #start = torch.tensor((0, 0, 0), device=device)
        #end = torch.tensor((5, 5, 5), device=device)

        #t = torch.linspace(0, 1, steps=100, device=device).unsqueeze(1)
        #bridge = start + t * (end - start)
#
        #z_bridge, _ = model(bridge, reverse=False)
#
        #bridge = bridge.cpu().numpy()
        #z_bridge = z_bridge.cpu().numpy()

        # Row 1
        axs[0,0].scatter(z_np[:,0], z_np[:,1], z_np[:,2], s=1, alpha=0.5)
        axs[0,0].set_title("Latent Space Samples")
        axs[0,0].axis("equal")

        axs[0,1].scatter(x_np[:,0], x_np[:,1], x_np[:,2], s=1, alpha=0.5)
        axs[0,1].set_title("Original Data")
        axs[0,1].axis("equal")

        #sample_data from models latent distribution
        z_dist = model.get_latent_distribution()
        z_samples = z_dist.sample((5000,))
        x_distribution_samples, _ = model(z_samples, reverse=True)
        x_distribution_samples = x_distribution_samples.cpu().numpy()
        z_samples = z_samples.cpu().numpy()
        
        axs[1,0].scatter(z_samples[:,0], z_samples[:,1], z_samples[:,2], s=1, alpha=0.5)
        axs[1,0].set_title("Latent Distribution Samples")
        axs[1,0].axis("equal")

        axs[1,1].scatter(x_distribution_samples[:,0], x_distribution_samples[:,1], x_distribution_samples[:,2], s=1, alpha=0.5)
        axs[1,1].set_title("Model Distribution Samples")
        axs[1,1].axis("equal")
        #Plot what the model thinks the data distribution looks like

        for ax in axs.flat:
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            ax.set_zlabel("dim 3")

    if save_path:
        plt.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=dpi)

    if show:
        plt.show()


def full_report_plot(model, data, device, save_path=None, dpi = 300, file_name = "full_report", show=False, line=False, z_line = None):

    z_dist = model.get_latent_distribution()

    fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(12, 18))

    model.eval()
    with torch.no_grad():
        grid = torch.meshgrid(torch.linspace(-15, 15, 1000), torch.linspace(-15, 15, 1000),indexing='xy')
        x = torch.stack(grid, dim=-1).reshape(-1, 2).to(device)
        #Get bridge points in original space
        start = torch.tensor((0, 0), device=device)
        end = torch.tensor((5, 5), device=device)

        t = torch.linspace(0, 1, steps=100, device=device).unsqueeze(1)
        bridge = start + t * (end - start)  
        z_bridge, _ = model(bridge, reverse=False)
        # First row: scatter plots


       # Right: transformed samples in latent space
        x_data = torch.tensor(data, dtype=torch.float32).to(device)
        z_data, _ = model(x_data, reverse=False)
        z_data = z_data.cpu().numpy()
        axs[0, 0].scatter(z_data[:, 0], z_data[:, 1], alpha=0.5, s=1)
        if line:
            axs[0, 0].plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=2, label='Bridge', color='r')
        axs[0, 0].set_title("Transformed Samples in Latent Space")
        axs[0, 0].set_xlabel("z[0]")
        axs[0, 0].set_ylabel("z[1]")
        axs[0, 0].set_aspect('equal')
        axs[0, 0].grid()

        # Left: original data distribution
        axs[0, 1].scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
        if line:
            axs[0, 1].plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=2, label='Bridge', color='r')
        axs[0, 1].set_title("Original Data Distribution")
        axs[0, 1].set_xlabel("x[0]")
        axs[0, 1].set_ylabel("x[1]")
        axs[0, 1].set_aspect('equal')
        axs[0, 1].grid()

 

        # Log-determinant of Jacobian for f (forward)
        _, ldj_f = model(x, reverse=False)
        ldj_f = ldj_f.cpu().numpy()
        axs[2, 1].imshow(ldj_f.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[2, 1].plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=2, label='Bridge', color='r')
        axs[2, 1].set_title("Log-Determinant Jacobian f(x)")
        axs[2, 1].set_xlabel("x[0]")
        axs[2, 1].set_ylabel("x[1]")
        axs[2, 1].grid()

        # Log-determinant of Jacobian for f^-1 (inverse)
        _, ldj_inv = model(x, reverse=True)
        ldj_inv = ldj_inv.cpu().numpy()
        axs[2, 0].imshow(ldj_inv.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[2, 0].plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=2, label='Bridge', color='r')
        axs[2, 0].set_title("Log-Determinant Jacobian f^-1(z)")
        axs[2, 0].set_xlabel("z[0]")
        axs[2, 0].set_ylabel("z[1]")
        axs[2, 0].grid()

        # Probability density heatmap
        z, ldj = model(x, reverse=False)
        log_qz = z_dist.log_prob(z)
        log_qx = ldj + log_qz
        qx = torch.exp(log_qx).cpu().numpy()

        axs[1, 1].imshow(qx.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[1, 1].plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=2, label='Bridge', color='r')
        axs[1, 1].set_title("Estimated Probability Density Heatmap")
        axs[1, 1].set_xlabel("x[0]")
        axs[1, 1].set_ylabel("x[1]")
        axs[1, 1].grid()

        #plot how the dataspace maps to the latent space
        # x and z has switched place here but it is just a semantic difference.
        z, ldj = model(x, reverse=True)
        log_qz = model.target_dist.log_prob(z)
        log_qx = ldj + log_qz
        qx = torch.exp(log_qx).cpu().numpy()

        axs[1, 0].imshow(qx.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[1, 0].plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=1, label='Bridge', color='r')
        axs[1, 0].set_title("Data Space Mapped to Latent Space")
        axs[1, 0].set_xlabel("z[0]")
        axs[1, 0].set_ylabel("z[1]")
        axs[1, 0].grid()

        probs = z_dist.log_prob(x)
        probs = torch.exp(probs).cpu().numpy()
        axs[3, 0].imshow(probs.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[3, 0].plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=1, label='Bridge', color='r')
        axs[3, 0].set_title("Latent distribution")
        axs[3, 0].set_xlabel("z[0]")
        axs[3, 0].set_ylabel("z[1]")
        axs[3, 0].grid()


        probs = model.target_dist.log_prob(x)
        probs = torch.exp(probs).cpu().numpy()
        axs[3, 1].imshow(probs.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[3, 1].plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=2, label='Bridge', color='r')
        axs[3, 1].set_title("Data distribution")
        axs[3, 1].set_xlabel("z[0]")
        axs[3, 1].set_ylabel("z[1]")
        axs[3, 1].grid()

    if save_path:
        plt.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=dpi)
    if show:
        plt.show()
    

def plot_distributions(model, device, line=True):
    z_dist = model.get_latent_distribution()
    model.eval()
    with torch.no_grad():
        # create a line of points passing through (0,1), (5,3.5) in the original space and see how it maps to the latent space
        start = torch.tensor((0, 0), device=device)
        end = torch.tensor((5, 5), device=device)
        t = torch.linspace(0, 1, steps=100, device=device).unsqueeze(1)
        bridge = start + t * (end - start)  
        z_bridge, _ = model(bridge, reverse=False)

        grid = torch.meshgrid(torch.linspace(-15, 15, 1000), torch.linspace(-15, 15, 1000),indexing='xy')
        x = torch.stack(grid, dim=-1).reshape(-1, 2).to(device)

        z, ldj = model(x, reverse=False)
        log_qz = z_dist.log_prob(z)
        log_qx = ldj + log_qz
        qx = torch.exp(log_qx).cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(qx.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            plt.plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=1, label='Bridge', color='r')
        plt.colorbar(label='Probability Density')
        plt.title("Estimated Probability Density Heatmap")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.grid()
    plt.show()
    plt.savefig("estimated_density.png", dpi=300)


def plot_13(model, device, save_path=None, dpi = 300, file_name = "full_report", show=False, line=False, z_line = None):
    z_dist = model.get_latent_distribution()

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), constrained_layout=True)

    model.eval()
    with torch.no_grad():
        rigth_grid = torch.meshgrid(torch.linspace(-15, 15, 1000), torch.linspace(-15, 15, 1000),indexing='xy')
        left_grid = torch.meshgrid(torch.linspace(-15, 15, 1000), torch.linspace(-15, 15, 1000),indexing='xy')

        x_rigth = torch.stack(rigth_grid, dim=-1).reshape(-1, 2).to(device)
        x_left  = torch.stack(left_grid, dim=-1).reshape(-1, 2).to(device)

        #Get bridge points in original space
        start = torch.tensor((0, 0), device=device)
        end = torch.tensor((5, 5), device=device)

        t = torch.linspace(0, 1, steps=100, device=device).unsqueeze(1)
        bridge = start + t * (end - start)  
        z_bridge, _ = model(bridge, reverse=False)
 


        # Probability density heatmap
        z, ldj = model(x_rigth, reverse=False)
        log_qz = z_dist.log_prob(z)
        log_qx = ldj + log_qz
        qx = torch.exp(log_qx).cpu().numpy()

        axs[1, 1].imshow(qx.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[1, 1].plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=2, label='Bridge', color='r')
        axs[1, 1].set_title("Estimated Probability Density Heatmap")
        axs[1, 1].set_xlabel("x[0]")
        axs[1, 1].set_ylabel("x[1]")
        axs[1, 1].grid()

        #plot how the dataspace maps to the latent space
        # x and z has switched place here but it is just a semantic difference.
        z, ldj = model(x_left, reverse=True)
        log_qz = model.target_dist.log_prob(z)
        log_qx = ldj + log_qz
        qx = torch.exp(log_qx).cpu().numpy()

        axs[1, 0].imshow(qx.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            axs[1, 0].plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=1, label='Bridge', color='r')
        axs[1, 0].set_title("Data Space Mapped to Latent Space")
        axs[1, 0].set_xlabel("z[0]")
        axs[1, 0].set_ylabel("z[1]")
        axs[1, 0].grid()

        probs = z_dist.log_prob(x_left)
        probs = torch.exp(probs).cpu().numpy()
        axs[0, 0].imshow(probs.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            pass
            #axs[0, 0].plot(z_bridge[:, 0], z_bridge[:, 1], 'r-', linewidth=1, label='Bridge', color='r')
        axs[0, 0].set_title("Latent distribution")
        axs[0, 0].set_xlabel("z[0]")
        axs[0, 0].set_ylabel("z[1]")
        axs[0, 0].grid()


        probs = model.target_dist.log_prob(x_rigth)
        probs = torch.exp(probs).cpu().numpy()
        axs[0, 1].imshow(probs.reshape(1000, 1000), extent=(-15, 15, -15, 15), origin='lower', cmap='viridis')
        if line:
            pass
            #axs[0, 1].plot(bridge[:, 0].cpu(), bridge[:, 1].cpu(), 'r-', linewidth=2, label='Bridge', color='r')
        axs[0, 1].set_title("Data distribution")
        axs[0, 1].set_xlabel("x[0]")
        axs[0, 1].set_ylabel("x[1]")
        axs[0, 1].grid()

    if save_path:
        plt.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=dpi)
    if show:
        plt.show()
    

if __name__ == "__main__":
    device = "cpu"
    print(f"Using device: {device}")

    # Load the trained model
    model_path = "/Users/eliasguedra/Documents/GitHub/ms-in-dnns/Report/TONF/Interesting runs/LearnablePriorNF2D_DIM=2_MVgaussian_2modal__lr=0.001_seed3735928558_20260312_181619_DEMO_2D/epoch=29-val_loss=3.54-standard_data-nobridge.ckpt"
    model_dir = os.path.dirname(model_path)
    model = LearnablePriorNFND.load_from_checkpoint(model_path)

    model.to(device)

    # Generate data from the target distribution
    num_samples = 5000
    data = model.target_dist.sample((num_samples,)).cpu().numpy()
    # Plot the transformed samples in latent space
    #plot_model(model, data, device)
    #plot_ljd_heatmap(model, device)
    #plot_probability_density(model, device)
    #plot_warping(model, device)
    #full_report_plot(model, data, device, save_path=model_dir, show=True)
    #full_report_plot_3d(model, data, device, save_path=model_dir, show=True)
    #plot_distributions(model, device)
    plot_13(model, device, save_path=model_dir, show=True, line=False)

    