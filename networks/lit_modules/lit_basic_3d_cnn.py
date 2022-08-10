
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

from .. import Basic3DCNN


class LitBasic3DCNN(pl.LightningModule):
	def __init__(
		self, 
		num_classes: int = 4, 
		overfit=False,
	):
		super().__init__()
		self.model = Basic3DCNN(num_classes=num_classes)
		self.overfit = overfit
		self.train_acc = torchmetrics.Accuracy()
		self.val_acc = torchmetrics.Accuracy()


	def forward(self, x):
		x = self.model(x)
		return x


	def training_step(self, train_batch, batch_idx):
		loss = self.step(train_batch, batch_idx, self.train_acc)
		self.log('loss/train', loss, sync_dist=True)
		self.log('acc/train', self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
		return loss

	def validation_step(self, val_batch, batch_idx):
		loss = self.step(val_batch, batch_idx, self.val_acc)
		self.log('loss/val', loss, sync_dist=True)
		self.log('acc/val', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


	def step(self, batch, batch_idx, acc_metric):
		x, y = batch
		y = y.flatten()
		out = self(x)
		loss = F.cross_entropy(out, y)
		acc_metric(out, y)
		return loss
    
