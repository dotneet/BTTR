from pytorch_lightning.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

def main():
    cli = LightningCLI(LitBTTR, CROHMEDatamodule, save_config_kwargs={"overwrite": True})

if __name__ == '__main__':
    main()
