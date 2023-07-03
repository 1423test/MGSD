import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import  AGNNConv,HeteroConv,Linear,GCNConv, SAGEConv, GATConv,RGCNConv
import torch
import torch.nn as nn
import random

class AGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        # self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self,x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training)
        # x = self.lin2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=True)

    def forward(self, x, edge_index,):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index,):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels/8, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels, hidden_channels,heads=1, concat=False,dropout=0.6)

    def forward(self, x, edge_index,):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class RGCN(torch.nn.Module):
    def __init__(self, graph,in_channels, hidden_channels,out_channels):
        super().__init__()
        self.conv1 = RGCNConv(graph.num_nodes,in_channels, hidden_channels,2)
        self.conv2 = RGCNConv(graph.num_nodes,hidden_channels, out_channels,2)

    def forward(self, x, edge_index,edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.normalize(x)

class heto(torch.nn.Module):
    def __init__(self, metadata,in_channels, hidden_channels,out_channels):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: AGNN (in_channels, hidden_channels)
            for edge_type in metadata[1]
            }, aggr='sum')

        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = self.lin2(x['nodes'])
        return F.normalize(x)


class diffusion(torch.nn.Module):

    def __init__(self,  diffusion_layer, lambda_,  noise, device):
        super(diffusion, self).__init__()
        self.diffusion_layer = diffusion_layer
        self.lambda_ = lambda_
        self.noise = noise
        self.device = device

    def edge_index2sparse_tensor(self, edge_index, node_num):
        sizes = (node_num, node_num)
        v = torch.ones(edge_index[0].numel()).to(self.device)
        return torch.sparse_coo_tensor(edge_index, v, sizes)

    def adj(self,edge_index, num_nodes):

        adj = self.edge_index2sparse_tensor(edge_index, num_nodes)
        degree_vector = torch.sparse.sum(adj, 0)
        degree_vector = degree_vector.to_dense().cpu()
        degree_vector = np.power(degree_vector, -0.5)
        degree_matrix = torch.diag(degree_vector).to(self.device)
        adj = torch.sparse.mm(adj.t(), degree_matrix.t())
        adj = adj.t()
        adj = torch.mm(adj, degree_matrix)
        adj = adj.to_sparse()

        return adj


    def augment(self, Z, Y, mask, edge_index,beta):

        num_nodes = Z.size()[0]
        G = Z.clone()
        G[mask] = Y.float()
        if self.noise == True:
            G1 = Z.clone()
            G1[mask] = Y.float()
            random.shuffle(mask)
            G[mask] = G1[mask] * beta + G[mask] * (1 - beta)
        else:
            G[mask] =  G[mask]

        adj = self.adj(edge_index, num_nodes)
        x = G.clone()

        for k in range(self.diffusion_layer):
            x = torch.sparse.mm(adj, x)
            x = x * self.lambda_
            x = x + (1 - self.lambda_) * G
        return x

