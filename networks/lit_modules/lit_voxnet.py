
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .. import VoxNet

class LitVoxNet(pl.LightningModule):
	def __init__(self, num_classes, overfit=False):
		super().__init__()
		self.model = VoxNet(num_classes=num_classes)
		self.overfit = overfit

	def forward(self, x):
		x = self.model(x)
		return x

	def configure_optimizers(self):
		if not self.overfit:
			# add l2 regularization/weight decay
			optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
		else:
			optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		loss = self.step(train_batch)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		loss = self.step(val_batch)
		self.log('val_loss', loss)

	def step(self, batch):
		x, y = batch
		out = self(x)
		loss = F.cross_entropy(out, y.flatten())
		return loss
    
