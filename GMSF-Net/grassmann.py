import math
import torch
from torch import nn
from torch.autograd import Function

dtype = torch.double
device = torch.device('cpu')

class DynamicColumnPartitionWithProjectionLayer(nn.Module):
    def __init__(self, n, max_channels):
        
        super(DynamicColumnPartitionWithProjectionLayer, self).__init__()
        self.n = n
        self.max_channels = max_channels

        self.partition_weights = nn.Parameter(torch.randn(max_channels, 15))  

    def forward(self, X):
        
        batch_size, _, row, col = X.shape

        partition_weights = torch.sigmoid(self.partition_weights)
        column_indices = self.select_columns(partition_weights)
        partitioned_columns = []
        for i in range(self.max_channels):
            selected_columns = X[:, :, :, column_indices[i]]  
            selected_weights = partition_weights[i, column_indices[i]].unsqueeze(0).unsqueeze(1)  
            weighted_columns = selected_columns * selected_weights

            partitioned_columns.append(weighted_columns)
        partitioned_columns = torch.cat(partitioned_columns, dim=1)
        return partitioned_columns

    def select_columns(self, partition_weights):

        weighted_col_indices = partition_weights 
        cluster_indices = torch.argsort(weighted_col_indices, dim=1)[:, :self.n]

        return cluster_indices


def calcuK(S):
    b, c, h = S.shape
    Sr = S.reshape(b, c, 1, h)
    Sc = S.reshape(b, c, h, 1)
    K = Sc - Sr
    K = 1.0 / K
    K[torch.isinf(K)] = 0
    K[torch.isnan(K)] = 0
    return K

def grassfusion(temp, p, alpha=1.0):
    batch, channel, k, n = temp.shape
    proj_layer = Projmap()
    log_maps = proj_layer(temp) 
    G_tangent = log_maps.mean(dim=1, keepdim=True)
    orth_layer = Orthmap(p)
    G_new = orth_layer(G_tangent) 
    G_new = alpha * G_new
    return G_new[:,0,:,:]


class FRMap(nn.Module):
    def __init__(self, out_channel,in_channel,input_size, output_size):
        super(FRMap, self).__init__()
        self.weight = nn.Parameter(torch.rand(out_channel,1,input_size, output_size, dtype=torch.double) * 2 - 1.0)
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.output = output_size

    def forward(self, x):
        weight, _ = torch.linalg.qr(self.weight)
        batch_size, channels_in, n_in, n_out = x.shape
        output = torch.zeros(batch_size, self.out_channel, self.output, n_out, dtype=x.dtype, device=x.device)
        for co in range(self.out_channel):
            temp = torch.zeros(batch_size, self.in_channel, self.output, n_out, dtype=x.dtype, device=x.device)
            for ci in range(self.in_channel):
                w = weight[co, 0].transpose(0, 1).unsqueeze(0)
                x_ci = x[:, ci]  # [32, 63, 10]
                temp[:,ci,:,:] = torch.matmul(w, x_ci)
            output[: ,co, :, : ]=grassfusion(temp,10)

        return output


class QRComposition(nn.Module):
    def __init__(self):
        super(QRComposition, self).__init__()

    def forward(self, x):
        Q, R = torch.linalg.qr(x)
        # flipping
        output = torch.matmul(Q, torch.diag_embed(torch.sign(torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        return output


class Projmap(nn.Module):
    def __init__(self):
        super(Projmap, self).__init__()

    def forward(self, x):
        return torch.matmul(x, x.transpose(-1, -2))


class Orthmap(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return OrthmapFunction.apply(x, self.p)


class OrthmapFunction(Function):
    @staticmethod
    def forward(ctx, x, p):
        U, S, V = torch.linalg.svd(x)
        ctx.save_for_backward(U, S)
        res = U[..., :p]
        return res

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        b, c, h, w = grad_output.shape
        p = h - w
        pad_zero = torch.zeros(b, c, h, p)
        grad_output = torch.cat((grad_output, pad_zero), 3)
        Ut = U.transpose(-1, -2)
        K = calcuK(S)
        mid_1 = K.transpose(-1, -2) * torch.matmul(Ut, grad_output)
        mid_2 = torch.matmul(U, mid_1)
        return torch.matmul(mid_2, Ut), None


class ProjPoolLayer_A(torch.autograd.Function):
    # AProjPooling  c/n ==0
    @staticmethod
    def forward(ctx, x, n=4):
        b, c, h, w = x.shape
        ctx.save_for_backward(n)
        new_c = int(math.ceil(c / n))
        new_x = [x[:, i:i + n].mean(1) for i in range(0, c, n)]
        return torch.cat(new_x, 1).reshape(b, new_c, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.saved_variables
        return torch.repeat_interleave(grad_output / n, n, 1)


class ProjPoolLayer(nn.Module):
    """ W-ProjPooling"""

    def __init__(self, n=4):
        super().__init__()
        self.n = n

    def forward(self, x):
        avgpool = torch.nn.AvgPool2d(int(math.sqrt(self.n)))
        return avgpool(x)
