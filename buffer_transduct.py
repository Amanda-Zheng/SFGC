from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from utils_graphsaint import DataGraphSAINT
import logging
import sys
import datetime
import os
from tensorboardX import SummaryWriter
import deeprobust.graph.utils as utils
from itertools import repeat
from models.gat import GAT
from models.gcn import GCN
from models.sgc import SGC


def main(args):
    # random seed setting
    random.seed(args.seed_teacher)
    np.random.seed(args.seed_teacher)
    torch.manual_seed(args.seed_teacher)
    torch.cuda.manual_seed(args.seed_teacher)
    device = torch.device(args.device)
    logging.info('args = {}'.format(args))

    data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
    if args.dataset in data_graphsaint:
        data = DataGraphSAINT(args.dataset)
        data_full = data.data_full
    else:
        data_full = get_dataset(args.dataset)
        data = Transd2Ind(data_full)

    features, adj, labels = data.feat_full, data.adj_full, data.labels_full
    adj, features, labels = utils.to_tensor(adj, features, labels, device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)

    adj = adj_norm

    trajectories = []

    model_type = args.buffer_model_type

    for it in range(0, args.num_experts):
        logging.info(
            '======================== {} -th number of experts for {}-model_type=============================='.format(
                it, model_type))

        model_class = eval(model_type)

        model = model_class(nfeat=features.shape[1], nhid=args.teacher_hidden, dropout=args.teacher_dropout,
                            nlayers=args.teacher_nlayers,
                            nclass=data.nclass, device=device).to(device)
        print(model)

        model.initialize()

        model_parameters = list(model.parameters())

        if args.optim == 'Adam':
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_teacher, weight_decay=args.wd_teacher)
        elif args.optim == 'SGD':
            optimizer_model = torch.optim.SGD(model_parameters, lr=args.lr_teacher, momentum=args.mom_teacher,
                                              weight_decay=args.wd_teacher)

        timestamps = []

        timestamps.append([p.detach().cpu() for p in model.parameters()])

        best_val_acc = best_test_acc = best_it = 0



        if args.dataset!='citeseer':
            lr_schedule = [args.teacher_epochs // 2 + 1]
        else:
            lr_schedule = [600]


        #lr_schedule = [args.teacher_epochs // 2 + 1]
        lr = args.lr_teacher
        for e in range(args.teacher_epochs + 1):
            model.train()
            optimizer_model.zero_grad()
            _,output = model.forward(features, adj)
            loss_buffer = F.nll_loss(output[data.idx_train], labels[data.idx_train])
            acc_buffer = utils.accuracy(output[data.idx_train], labels[data.idx_train])
            writer.add_scalar('buffer_train_loss_curve', loss_buffer.item(), e)
            writer.add_scalar('buffer_train_acc_curve', acc_buffer.item(), e)
            logging.info("Epochs: {} : Full graph train set results: loss= {:.4f}, accuracy= {:.4f} ".format(e,
                                                                                                             loss_buffer.item(),
                                                                                                             acc_buffer.item()))
            loss_buffer.backward()
            optimizer_model.step()

            if e in lr_schedule and args.decay:
                lr = lr*args.decay_factor
                logging.info('NOTE! Decaying lr to :{}'.format(lr))
                if args.optim == 'SGD':
                    optimizer_model = torch.optim.SGD(model_parameters, lr=lr, momentum=args.mom_teacher,weight_decay=args.wd_teacher)
                elif args.optim == 'Adam':
                    optimizer_model = torch.optim.Adam(model_parameters, lr=lr,
                                                       weight_decay=args.wd_teacher)

                optimizer_model.zero_grad()

            if e % 20 == 0:
                logging.info("Epochs: {} : Train set training:, loss= {:.4f}".format(e, loss_buffer.item()))
                model.eval()
                labels_val = torch.LongTensor(data.labels_val).cuda()
                labels_test = torch.LongTensor(data.labels_test).cuda()

                # Full graph
                _,output = model.predict(data.feat_full, data.adj_full)
                loss_val=F.nll_loss(output[data.idx_val], labels_val)
                loss_test = F.nll_loss(output[data.idx_test], labels_test)

                acc_val = utils.accuracy(output[data.idx_val], labels_val)
                acc_test = utils.accuracy(output[data.idx_test], labels_test)

                writer.add_scalar('val_set_loss_curve', loss_val.item(), e)
                writer.add_scalar('val_set_acc_curve', acc_val.item(), e)

                writer.add_scalar('test_set_loss_curve', loss_test.item(), e)
                writer.add_scalar('test_set_acc_curve', acc_test.item(), e)

                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_it = e

            if e % args.param_save_interval == 0 and e>1:
                timestamps.append([p.detach().cpu() for p in model.parameters()])
                p_current = timestamps[-1]
                p_0 = timestamps[0]
                target_params = torch.cat([p_c.data.reshape(-1) for p_c in p_current], 0)
                starting_params = torch.cat([p0.data.reshape(-1) for p0 in p_0], 0)
                param_dist1 = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                writer.add_scalar('param_change', param_dist1.item(), e)
                logging.info(
                    '==============================={}-th iter with length of {}-th tsp'.format(e, len(timestamps)))

        logging.info("Valid set best results: accuracy= {:.4f}".format(best_val_acc.item()))
        logging.info("Test set best results: accuracy= {:.4f} within best iteration = {}".format(best_test_acc.item(),best_it))
        trajectories.append(timestamps)

        if len(trajectories) == args.traj_save_interval:
            n = 0
            while os.path.exists(os.path.join(log_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            logging.info("Saving {}".format(os.path.join(log_dir, "replay_buffer_{}.pt".format(n))))
            #torch.save(trajectories, os.path.join(log_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []




class GraphData:

    def __init__(self, features, adj, labels, idx_train=None, idx_val=None, idx_test=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


from torch_geometric.data import Data
from models.in_memory_dataset import InMemoryDataset
import scipy.sparse as sp


class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/'  # dummy root; does not mean anything
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--teacher_epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('--teacher_nlayers', type=int, default=2)
    parser.add_argument('--teacher_hidden', type=int, default=256)
    parser.add_argument('--teacher_dropout', type=float, default=0.0)
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for buffer learning rate')
    parser.add_argument('--wd_teacher', type=float, default=0)
    parser.add_argument('--mom_teacher', type=float, default=0)
    parser.add_argument('--seed_teacher', type=int, default=15, help='Random seed.')
    parser.add_argument('--num_experts', type=int, default=200, help='training iterations')
    parser.add_argument('--param_save_interval', type=int, default=10)
    parser.add_argument('--traj_save_interval', type=int, default=10)
    parser.add_argument('--save_log', type=str, default='logs', help='path to save logs')
    parser.add_argument('--buffer_model_type', type=str, default='GCN', help='Default buffer_model type')
    parser.add_argument('--optim', type=str, default='SGD', choices=['Adam', 'SGD'], help='Default buffer_model type')
    parser.add_argument('--decay', type=int, default=0, choices=[1, 0], help='whether to decay lr at 1/2 training epochs')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor of lr at 1/2 training epochs')

    args = parser.parse_args()

    log_dir = './' + args.save_log + '/Buffer/{}-{}'.format(args.dataset,
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    writer = SummaryWriter(log_dir + '/tbx_log')
    main(args)
    logging.info(args)
    logging.info('Finish!, Log_dir: {}'.format(log_dir))
