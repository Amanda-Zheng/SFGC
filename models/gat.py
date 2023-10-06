import torch.nn.functional as F
import torch
from .mygatconv import GATConv
from itertools import repeat


class GAT(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, with_bias=True, device=None, **kwargs):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout

        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

        self.output = None
        self.best_model = None
        self.best_output = None
        self.initialize()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data)



class GraphData:

    def __init__(self, features, adj, labels, idx_train=None, idx_val=None, idx_test=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


from torch_geometric.data import Data
from .in_memory_dataset import InMemoryDataset
import scipy.sparse as sp

class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data,transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process____(self):
        dpr_data = self.dpr_data
        try:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero().cpu()).cuda().T
        except:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero()).cuda()
        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def process(self):
        dpr_data = self.dpr_data
        if type(dpr_data.adj) == torch.Tensor:
            adj_selfloop = dpr_data.adj + torch.eye(dpr_data.adj.shape[0]).cuda()
            edge_index_selfloop = adj_selfloop.nonzero().T
            edge_index = edge_index_selfloop
            edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
        else:
            adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
            edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
            edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()
        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels).cuda()
        except:
            y = dpr_data.labels

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]

        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

