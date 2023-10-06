import random
import argparse
# from configs import load_config
from utils import *
from utils_graphsaint import DataGraphSAINT
from models.gcn import GCN
from coreset import KCenter, Herding, Random
from tqdm import tqdm
import torch
import deeprobust.graph.utils as utils
import datetime
import os
import sys
import torch.nn.functional as F

parser = argparse.ArgumentParser()
# parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'reddit'])
parser.add_argument('--hidden', type=float, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr_coreset', type=float, default=0.01)
parser.add_argument('--wd_coreset', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_log', type=str, default='logs')
parser.add_argument('--method', type=str, default='kcenter', choices=['kcenter', 'herding', 'random'])
parser.add_argument('--reduction_rate', type=float, default=0.5)
parser.add_argument('--load_npy', type=str, default='')
parser.add_argument('--opt_type_train', type=str, default='Adam')
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

device = torch.device(args.device)
# torch.cuda.set_device(args.gpu_id)
# args = load_config(args)
# print(args)
log_dir = './' + args.save_log + '/Coreset/{}-reduce_{}-{}'.format(args.dataset, str(args.reduction_rate),
                                                                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'coreset.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('This is the log_dir: {}'.format(log_dir))

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data_graphsaint = ['flickr', 'reddit']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

features, adj, labels = data.feat_train, data.adj_train, data.labels_train
adj, features, labels = utils.to_tensor(adj, features, labels, device=device)
# labels = torch.FloatTensor(labels)
adj, features, labels = adj.to(device), features.to(device), labels.to(device)
feat_test, adj_test, labels_test = data.feat_test, data.adj_test, data.labels_test
adj_test, feat_test, labels_test = utils.to_tensor(adj_test, feat_test, labels_test, device=device)

feat_val, adj_val, labels_val = data.feat_val, data.adj_val, data.labels_val
adj_val, feat_val, labels_val = utils.to_tensor(adj_val, feat_val, labels_val, device=device)

if utils.is_sparse_tensor(adj):
    adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
else:
    adj_norm = utils.normalize_adj_tensor(adj)

adj = adj_norm.to(device)

# Setup GCN Model
# device = 'cuda'
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=data.nclass, device=device,
            weight_decay=args.weight_decay)

model = model.to(device)
if args.load_npy == '':
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_load_test = best_load_it = 0
    for e in range(args.epochs + 1):
        model.train()
        optimizer_model.zero_grad()
        embed, output = model.forward(features, adj)
        loss = F.nll_loss(output, labels)
        acc = utils.accuracy(output, labels)

        # print('Epochs:', e, 'Full graph train set results: loss = ',loss.item(), 'accuracy=',acc.item())
        if e % 10 == 0:
            logging.info('=========Train===============')
            logging.info(
                'Epochs={}: Full graph train set results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss.item(),
                                                                                                   acc.item()))
        loss.backward()
        optimizer_model.step()
        if e % 1 == 0:
            model.eval()
            # You can use the inner function of model to test
            _, output_test = model.predict(feat_test, adj_test)
            loss_test = F.nll_loss(output_test, labels_test)
            acc_test = utils.accuracy(output_test, labels_test)
            if acc_test > best_load_test:
                best_load_test = acc_test.item()
                best_load_it = e
                # print('=============Testing===============')
                logging.info('=========Testing===============')
                # print('Epochs:', e, 'Test results: loss = ',loss_test.item(), 'accuracy=',acc_test.item())

        embed_out = embed
    logging.info(
        'BEST Test results: accuracy = {:.4f} within {}-th iterations'.format(best_load_test, best_load_it))

    if args.method == 'kcenter':
        agent = KCenter(data, args, device=device)
    if args.method == 'herding':
        agent = Herding(data, args, device=device)
    if args.method == 'random':
        agent = Random(data, args, device=device)
    idx_selected = agent.select(embed_out, inductive=True)
    feat_train = features[idx_selected]
    adj_train = data.adj_train[np.ix_(idx_selected, idx_selected)]
    labels_train = labels[idx_selected]
    if args.save:
        logging.info('Saving...')
        np.save(f'{log_dir}/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy', idx_selected)
    logging.info(args)
    logging.info(log_dir)
else:
    res = []
    # runs = 10
    logging.info('Loading from: {}'.format(args.load_npy))
    idx_selected_train = np.load(
        f'{args.load_npy}/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy')
    feat_train = features[idx_selected_train]
    # feat_train = F.normalize(feat_train,p=1)
    adj_train = data.adj_train[np.ix_(idx_selected_train, idx_selected_train)]
    # adj_train = torch.ones((feat_train.shape[0],feat_train.shape[0]))
    # adj_train = torch.eye(feat_train.shape[0])
    labels_train = labels[idx_selected_train]
    if sp.issparse(adj_train):
        adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)
    else:
        adj_train = torch.FloatTensor(adj_train)
    adj_train, feat_train, labels_train = adj_train.to(device), feat_train.to(device), labels_train.to(device)
    # assert False

    if utils.is_sparse_tensor(adj_train):
        adj_train_norm = utils.normalize_adj_tensor(adj_train, sparse=True)
    else:
        adj_train_norm = utils.normalize_adj_tensor(adj_train)

    adj_train = adj_train_norm.to(device)
    if args.opt_type_train == 'Adam':
        optimizer_model_coreset = torch.optim.Adam(model.parameters(), lr=args.lr_coreset, weight_decay=args.wd_coreset)
    elif args.opt_type_train == 'SGD':
        optimizer_model_coreset = torch.optim.SGD(model.parameters(), lr=args.lr_coreset, momentum=0.9)
    for _ in tqdm(range(args.runs)):
        model.initialize()
        best_test_acc = 0
        for e in range(args.epochs + 1):
            model.train()
            optimizer_model_coreset.zero_grad()
            _, output_train = model.forward(feat_train, adj_train)
            loss_train = F.nll_loss(output_train, labels_train)
            acc_train = utils.accuracy(output_train, labels_train)
            logging.info('=========Train coreset===============')
            logging.info('Epochs={}: coreset results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_train.item(),
                                                                                               acc_train.item()))

            # print('Epochs:', e, 'Train graph set results: loss = ', loss_train.item(), 'accuracy=', acc_train.item())
            loss_train.backward()
            optimizer_model_coreset.step()
            if e % 10 == 0:
                model.eval()
                # You can use the inner function of model to test
                _, output_test = model.predict(feat_test, adj_test)
                loss_test = F.nll_loss(output_test, labels_test)
                acc_test = utils.accuracy(output_test, labels_test)
                if acc_test > best_test_acc:
                    best_test_acc = acc_test.item()
                    best_it = e
                logging.info('=========Test coreset===============')
                logging.info(
                    'Epochs={}: Test coreset results: loss = {:.4f}, accuracy = {:.4f} within the best_acc = {}, best-iter = {}'.format(
                        e, loss_test.item(),
                        acc_test.item(), best_test_acc, best_it))
        res.append(best_test_acc)

    #    model.fit_with_val(feat_train, adj_train, labels_train, data,
    #                 train_iters=600, normalize=True, verbose=False)
    #
    #    model.eval()
    #    labels_test = torch.LongTensor(data.labels_test).cuda()
    #
    #    # Full graph
    #    output = model.predict(data.feat_full, data.adj_full)
    #    loss_test = F.nll_loss(output[data.idx_test], labels_test)
    #    acc_test = utils.accuracy(output[data.idx_test], labels_test)
    #    res.append(acc_test.item())
    #
    res = np.array(res)
    logging.info(args)
    logging.info(log_dir)
    logging.info('Mean accuracy = {:.4f}, Std = {:.4f}'.format(res.mean(), res.std()))
