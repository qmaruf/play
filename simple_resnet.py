import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn as nn
from sklearn import metrics
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import numpy as np
from sklearn.metrics import precision_score
warnings.filterwarnings('ignore')


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outc)
    
    def forward(self, x):        
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.conv1 = ConvBlock(inc, inc, stride=1)
        self.conv2 = ConvBlock(inc, inc, stride=1)
        
    def forward(self, x):        
        return x + self.conv2(self.conv1(x))


class ConvResBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = ConvBlock(inc, outc)
        self.res = ResBlock(outc)

    def forward(self, x):
        x = self.conv(x)        
        x = self.res(x)
        return x

class SimpleNet(pl.LightningModule):
    def __init__(self, criterion):
        super(SimpleNet, self).__init__() 
        self.criterion = criterion
        self.crb1 = ConvResBlock(1, 8)
        self.crb2 = ConvResBlock(8, 16)
        self.crb3 = ConvResBlock(16, 32)
        self.crb4 = ConvResBlock(32, 16)
        self.crb5 = ConvResBlock(16, 10)
    
    def forward(self, x):       
        x = self.crb1(x)        
        x = self.crb2(x)        
        x = self.crb3(x)        
        x = self.crb4(x)
        x = self.crb5(x)
        x = x.view(x.size(0), -1)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):        
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.criterion(y_hat, y)}

    def validation_end(self, outputs):        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        
    def test_step(self, batch, batch_idx):        
        x, y = batch
        y_hat = self.forward(x)
        y_hat_lbl = torch.argmax(y_hat, dim=1)
        return {'test_loss': self.criterion(y_hat, y)}

    def test_end(self, outputs):        
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()  
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):        
        return torch.optim.SGD(self.parameters(), lr=0.1)

    @pl.data_loader
    def train_dataloader(self):        
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=128)

    @pl.data_loader
    def val_dataloader(self):        
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=128)

    @pl.data_loader
    def test_dataloader(self):        
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=128)

early_stop_callback = EarlyStopping( monitor='val_loss', patience=8, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(filepath='./checkpoint/', save_top_k=1, verbose=True, monitor='val_loss', mode='min', prefix='')

criterion = nn.CrossEntropyLoss()
model = SimpleNet(criterion=criterion)
trainer = Trainer(max_epochs=10,
	early_stop_callback=early_stop_callback,
	checkpoint_callback=checkpoint_callback)    
trainer.fit(model) 

model.eval()
y_pred, y_true = [], []
for data in model.test_dataloader():
    for X, y in data:
        pred = torch.argmax(model(X), dim=1)
        y_pred.append(pred.data.cpu().numpy())
        y_true.append(y.data.cpu().numpy())
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

precision = precision_score(y_true, y_pred, average='micro')
print (precision)