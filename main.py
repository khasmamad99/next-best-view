from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from networks.lit_modules import LitVoxNet, LitBasic3DCNN
from data.lit_modules import LitShapeNetNBV


def main():
    cli = LightningCLI(LitBasic3DCNN, LitShapeNetNBV)


if __name__ == "__main__":
    main()
