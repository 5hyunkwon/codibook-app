import yaml
import copy
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CodiClassifier
from loader import CodiDatasetLoader


class CodiTrain:
    def __init__(self, config_path):
        self.__set_configs(config_path)

        dataset_loader = CodiDatasetLoader(self.configs['image_path'])
        train_ds = dataset_loader('train')
        val_ds = dataset_loader('val')
        self.train_dl = DataLoader(train_ds, batch_size=self.configs['batch_size']['train'], shuffle=True)
        self.val_dl = DataLoader(val_ds, batch_size=self.configs['batch_size']['val'])

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        codi_classifier = CodiClassifier(self.configs)
        self.model = codi_classifier.get_model().to(self.device)
        self.weight_path = self.configs['weight_path']

        self.loss_func = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=self.configs['learning_rate'])
        self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.1, patience=10, verbose=True)
        self.num_epochs = self.configs['num_epochs']

    def __set_configs(self, config_path):
        with open(config_path) as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)

    def get_lr(self):
        for param_group in self.opt.param_groups:
            return param_group['lr']

    @staticmethod
    def metrics_batch(output, target):
        pred = output.argmax(dim=1, keepdim=True)
        corrects = pred.eq(target.view_as(pred)).sum().item()
        return corrects

    def loss_batch(self, output, target, opt=None):
        loss = self.loss_func(output, target)
        with torch.no_grad():
            metric_b = self.metrics_batch(output, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), metric_b

    def loss_epoch(self, dataset_dl, opt=None):
        running_loss = 0.0
        running_metric = 0.0
        len_data = len(dataset_dl.dataset)

        for xb, yb in tqdm(dataset_dl):
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            output = self.model(xb)
            loss_b, metric_b = self.loss_batch(output, yb, opt)

            running_loss += loss_b
            if metric_b is not None:
                running_metric += metric_b

        loss = running_loss / float(len_data)
        metric = running_metric / float(len_data)
        return loss, metric

    def train(self):
        loss_history = {
            'train': [],
            'val': []
        }

        metric_history = {
            'train': [],
            'val': []
        }

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            current_lr = self.get_lr()
            print(f'Epoch {epoch + 1}/{self.num_epochs}, current lr={current_lr}')

            self.model.train()
            train_loss, train_metric = self.loss_epoch(self.train_dl, self.opt)
            loss_history['train'].append(train_loss)
            metric_history['train'].append(train_metric)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_metric = self.loss_epoch(self.val_dl)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.weight_path)
                print('Copied best model weights!')

            self.lr_scheduler.step(val_loss)
            if current_lr != self.get_lr():
                print('Loading best model weights!')
                self.model.load_state_dict(best_model_wts)

            print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {(100 * val_metric):.2f}')
            print('-' * 10)

        self.model.load_state_dict(best_model_wts)
        return loss_history, metric_history


if __name__ == "__main__":
    codi_train = CodiTrain('./configs.yaml')
    loss_hist = codi_train.train()
