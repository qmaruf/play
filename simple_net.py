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

class SimpleNet(pl.LightningModule):

    def __init__(self, criterion):
        super(SimpleNet, self).__init__() 
        self.criterion = criterion       
        self.conv1 = self.make_layers(1, 8)
        self.conv2 = self.make_layers(8, 16)
        self.conv3 = self.make_layers(16, 32)
        self.conv4 = self.make_layers(32, 16)
        self.conv5 = self.make_layers(16, 10)        
        
    def make_layers(self, inc, outc):
        layers = []                
        layers += [nn.Conv2d(inc, outc, kernel_size=3, stride=2, padding=1)]
        layers += [nn.ReLU()]        
        layers += [nn.BatchNorm2d(outc)]            
        return nn.Sequential(*layers)

    def forward(self, x):       
        x = self.conv1(x)        
        x = self.conv2(x)        
        x = self.conv3(x)        
        x = self.conv4(x)
        x = self.conv5(x)
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