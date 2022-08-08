from typing import Optional
from pathlib import Path

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .. import ShapeNetNBV


class LitShapeNetNBV(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir_path: str = 'data/data/ShapeNetCore.v2_nbv',
        file_extension: str = 'pickle',
        batch_size: int = 32,
        overfit: bool = False,
        num_train_objs: int = 200,
        num_val_objs: int = 25,
        num_test_objs: int = 25,
    ):
        super().__init__()
        self.data_dir_path = Path(data_dir_path)
        self.file_extension = file_extension
        self.batch_size = batch_size
        self.overfit = overfit
        if overfit:
            # load only a single object
            self.train_objs_start, self.train_objs_end = 0, 1
            self.val_objs_start, self.val_objs_end = 0, 1
            self.test_objs_start, self.test_objs_end = 0, 1
        else:
            self.train_objs_start, self.train_objs_end = 0, num_train_objs
            self.val_objs_start, self.val_objs_end = self.train_objs_end, self.train_objs_end + num_val_objs
            self.test_objs_start, self.test_objs_end = self.val_objs_end, self.val_objs_end + num_test_objs


    def setup(self, stage: Optional[str] = None):
        self.train_set = ShapeNetNBV(
            self.data_dir_path, self.file_extension, self.train_objs_start, self.train_objs_end)
        self.val_set = ShapeNetNBV(
            self.data_dir_path, self.file_extension, self.val_objs_start, self.val_objs_end)
        self.test_set = ShapeNetNBV(
            self.data_dir_path, self.file_extension, self.test_objs_start, self.test_objs_end)


    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=1, persistent_workers=True)


    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=1, persistent_workers=True)

    
    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=1, persistent_workers=False)

        



