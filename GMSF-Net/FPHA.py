import sys
import os
sys.path.extend([sys.path[0]+'/..'])
import sys

sys.path.insert(0,'..')
from pathlib import Path
import os
import random

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data

import nn as nn_spd
from optimizers import MixOptimizer
from scipy.io import loadmat
from torch.utils import data
import grassmann as grass


def FPHA(data_loader):

    #main parameters
    lr=2e-3
    n=93 #dimension of the data
    C=117 #number of classes
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=200

    #setup data and model
    class FPHANet(nn.Module):
        def __init__(self):
            super(__class__,self).__init__()
            dim=63
            dim1=63
            classes=46
            self.batchnorm = nn_spd.BatchNormSPD(dim)
            self.frmap = grass.FRMap( 10,7,63, 36)
            self.ormap = grass.Orthmap(15)
            self.reorth = grass.QRComposition()
            self.projmap = grass.Projmap()
            self.partition_layer = grass.DynamicColumnPartitionWithProjectionLayer(n=10, max_channels=7)
            self.linear=nn.Linear(12960,classes).double()
        def forward(self,x):
            x = x.double()
            x = x.unsqueeze(1)
            x = self.batchnorm(x)
            x = self.ormap(x)
            subspace = self.partition_layer(x)
            x = self.frmap(subspace)
            x = self.projmap(x).view(x.shape[0],-1)
            y=self.linear(x)
            return y,subspace

    model=FPHANet()

    class MutualInformationLoss(nn.Module):
        def __init__(self, eps=1e-8):
            super(MutualInformationLoss, self).__init__()
            self.eps = eps

        def forward(self, X):
            """
            X: Tensor of shape (B, n, d1, d2)
            B = batch size
            n = number of Grassmannian subspaces
            d1, d2 = each subspace matrix dimensions
            """
            B, n, d1, d2 = X.shape

            X_flat = X.view(B, n, -1)  # shape: (B, n, d1*d2)

            X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)  # shape: (B, n, d1*d2)

            var = (X_centered ** 2).mean(dim=(0, 2))  # shape: (n,)

            cov = th.zeros(n, n, device=X.device)
            for i in range(n):
                Xi = X_centered[:, i, :]  # (B, d1*d2)
                for j in range(n):
                    Xj = X_centered[:, j, :]
                    cov[i, j] = (Xi * Xj).mean()

            corr = cov / th.sqrt(var[:, None] * var[None, :] + self.eps)

            # Mutual Information Loss
            loss = -th.log(th.abs(corr) + self.eps).sum()
            return loss
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_add = MutualInformationLoss()
    opti = MixOptimizer(model.parameters(),lr=lr)

    #initial validation accuracy
    loss_val,acc_val=[],[]
    max_temp = 0.0
    acc_valt = []
    y_true,y_pred=[],[]
    gen = data_loader._test_generator
    model.eval()
    for local_batch, local_labels in gen:
        out,subspace = model(local_batch)
        l = loss_fn(out, local_labels)
        predicted_labels=out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
        loss_val.append(loss)
        acc_val.append(acc)
    acc_val = np.asarray(acc_val).mean()
    loss_val = np.asarray(loss_val).mean()
    print('Initial validation accuracy: '+str(100*acc_val)+'%')

    #training loop
    for epoch in range(epochs):

        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        for local_batch, local_labels in data_loader._train_generator:
            opti.zero_grad()
            out,subspace = model(local_batch)
            l = loss_fn(out, local_labels)
            l1 = loss_fn_add(subspace)
            l = l + 0.01*l1
            acc, loss = (out.argmax(1)==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()

        # validation
        acc_val_list=[]
        y_true,y_pred=[],[]
        model.eval()
        for local_batch, local_labels in data_loader._test_generator:
            out,subspace = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels=out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            acc_val_list.append(acc)
        xx=None
        acc_val = np.asarray(acc_val_list).mean()
        max_temp = max(max_temp,acc_val)
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))

    print('Final validation accuracy: '+str(100*acc_val)+'%')
    return 100*acc_val,max_temp

if __name__ == "__main__":

    data_path='./data/FHPA/'
    pval=0.5
    batch_size=30 #batch size


    def is_spd(matrix, tol=1e-6):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
           return False

        if not th.allclose(matrix, matrix.T, atol=tol):
           return False

        try:
           eigvals = th.linalg.eigvalsh(matrix)
           return th.all(eigvals > tol)
        except RuntimeError:
           return False

    class CovarianceTransform:
        def __init__(self, scale=10.0, epsilon=8e-6, device='cpu'):
            self.scale = scale
            self.epsilon = epsilon
            self.device = device

        def __call__(self, sample):
            T = sample.to(self.device) * self.scale
            C, T_len = T.shape
            T_mean = T.mean(dim=1, keepdim=True)
            T_centered = T - T_mean
            T_cov = (T_centered @ T_centered.T) / (T_len - 1)
            trace_val = th.trace(T_cov)
            T_cov += trace_val * self.epsilon * th.eye(C, device=self.device)
            return T_cov

    class DatasetFPHA(data.Dataset):
        def __init__(self, seqs, labels, transform=None):
            self._seqs = seqs
            self._labels = labels
            self._transform = transform

        def __len__(self):
            return len(self._seqs)

        def __getitem__(self, item):
            x = self._seqs[item]
            y = int(self._labels[item])
            x = th.from_numpy(x).float()
            if self._transform:
                x = self._transform(x)
            y = th.tensor(y).long()
            return x, y


    class DataLoaderFPHA:
        def __init__(self, data_path, batch_size=32, device='cpu'):
            # 加载 .mat 文件
            train_seq = self._load_mat(data_path + '/FPHA_train_seq.mat')
            train_label = self._load_mat(data_path + '/FPHA_train_label.mat')
            val_seq = self._load_mat(data_path + '/FPHA_val_seq.mat')
            val_label = self._load_mat(data_path + '/FPHA_val_label.mat')

            # 初始化 transform
            transform = CovarianceTransform(device=device)

            # 构造 Dataset 和 DataLoader
            train_set = DatasetFPHA(train_seq, train_label, transform=transform)
            val_set = DatasetFPHA(val_seq, val_label, transform=transform)

            self._train_generator = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            self._test_generator = data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

        def _load_mat(self, filepath):
            mat = loadmat(filepath)
            key = [k for k in mat.keys() if not k.startswith("__")][0]
            data = mat[key]
            if isinstance(data[0], np.ndarray):
                return [d.astype(np.float32) for d in data[0]]
            else:
                return data.squeeze().astype(np.int64)

        @property
        def train_generator(self):
            return self._train_generator

        @property
        def test_generator(self):
            return self._test_generator


    loader = DataLoaderFPHA(data_path=data_path, batch_size=32, device='cpu')
    temp, maxtemp = FPHA(loader)

