from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

def main():
    cli = LightningCLI(LitBTTR, CROHMEDatamodule)

if __name__ == '__main__':
    main()
