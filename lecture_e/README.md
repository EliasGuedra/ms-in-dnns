# Lecture E: PyTorch Lightning

Relevant documentation:

## PyTorch Lightning
- [Lightning documentation](https://lightning.ai/docs/pytorch/2.1.2/)
- [`LigntningModule` documentation](https://lightning.ai/docs/pytorch/2.1.2/common/lightning_module.html)
- [`Trainer` documentation](https://lightning.ai/docs/pytorch/2.1.2/common/trainer.html)
- [`LightningDataModule` documentation](https://lightning.ai/docs/pytorch/2.1.2/data/datamodule.html)
- [Documentation on logging in lightning](https://lightning.ai/docs/pytorch/2.1.2/extensions/logging.html)
- [Callback documentation](https://lightning.ai/docs/pytorch/2.1.2/extensions/callbacks.html)

## Torchmetrics
- [Torchmetrics documentation](https://lightning.ai/docs/torchmetrics/stable/)
- [Interaction of torchmetrics and lightning](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
- [`MetricCollection` documentation](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection)

## WSL (Windows Subsystem for Linux)
For Task 2, Windows is often problematic when trying to install `escnn`, and the most convenient alternative is to use a Windows Subsystem for Linux (WSL).

To install, run in Powershell as the Administrator:
```bash
wsl --install
```

To enter the subsystem, run:
```bash
wsl
```

To install Python with a specific version (for example 3.10), run:
```bash
sudo apt-get install -y python3.10
```

For more information, check the official documentation [here](https://learn.microsoft.com/en-us/windows/wsl/install).

After installation, to use WSL in Visual Studio Code, go to `Extensions` in the sidebar, install the `WSL` extension. Press `Ctrl` + `Shift` + `P`, type "WSL" and select "WSL: Connect to WSL in New Window".

In the new window, you could open the same directory to the assignments, but with a different path format. Specifically, `"C:\path\to\ms-in-dnns"` in Windows corresponds to `"/mnt/c/path/to/ms-in-dnns"` in WSL. Note that the path delimiter is now "/" instead of "\". Similarly, you would need to build a new virtual environment that is compatible with WSL:
```bash
python3.10 -m venv venv_wsl
```
You could absolutely name it otherwise. To activate it, run:
```bash
source venv_wsl/bin/activate
```

Go through Assignment A.1 again to build the dependencies.

