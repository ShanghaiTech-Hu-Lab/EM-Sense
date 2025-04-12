
from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI




def run_cli():
    seed_everything(42, workers=True)
    LightningCLI(save_config_callback=None)
    

if __name__ == "__main__":
    run_cli()