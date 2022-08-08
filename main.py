from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from networks.lit_modules import LitVoxNet
from data.lit_modules import LitShapeNetNBV


def main():
    cli = LightningCLI(LitVoxNet, LitShapeNetNBV)


if __name__ == "__main__":
    main()
