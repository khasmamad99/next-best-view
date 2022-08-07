
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

from .. import VoxNet

class LitVoxNet(pl.LightningModule):
	def __init__(self, num_classes, overfit=False):
		super().__init__()
		self.model = VoxNet(num_classes=num_classes)
		self.overfit = overfit
		self.train_acc = torchmetrics.Accuracy()
		self.val_acc = torchmetrics.Accuracy()

	def forward(self, x):
		x = self.model(x)
		return x

	def configure_optimizers(self):
		if not self.overfit:
			# add l2 regularization/weight decay
			optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, 
				milestones=[10, 20, 30], 
				gamma=0.3, 
				verbose=False
			)
			return [optimizer], [lr_scheduler]
		else:
			optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
			return optimizer

	def training_step(self, train_batch, batch_idx):
		loss = self.step(train_batch, batch_idx, self.train_acc)
		self.log('train_loss', loss)
		self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
		return loss

	def validation_step(self, val_batch, batch_idx):
		loss = self.step(val_batch, batch_idx, self.val_acc)
		self.log('val_loss', loss)
		self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


	def step(self, batch, batch_idx, acc_metric):
		x, y = batch
		y = y.flatten()
		out = self(x)
		loss = F.cross_entropy(out, y)
		acc_metric(out, y)
		return loss
    
