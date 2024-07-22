import scipy.sparse as sp
import pandas as pd
import numpy as np
import random
import glob
import os
import sys
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from scGNN import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
from PytorchTools import EarlyStopping

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from torch_geometric.utils import negative_sampling
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel


parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=3e-3,
                    help='Initial learning rate.')
parser.add_argument('-epochs', type=int, default=10, help='Number of epoch.')
parser.add_argument('-num_head', type=list,
                    default=[3, 3], help='Number of head attentions.')
parser.add_argument('-alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('-hidden_dim', type=int,
                    default=[128, 64, 32], help='The dimension of hidden layer')
parser.add_argument('-output_dim', type=int, default=16,
                    help='The dimension of latent layer')
parser.add_argument('-batch_size', type=int, default=256,
                    help='The size of each batch')
parser.add_argument('-loop', type=bool, default=False,
                    help='whether to add self-loop in adjacent matrix')
parser.add_argument('-seed', type=int, default=42, help='Random seed')
parser.add_argument('-Type', type=str, default='dot', help='score metric')
parser.add_argument('-flag', type=bool, default=False,
                    help='the identifier whether to conduct causal inference')
parser.add_argument('-reduction', type=str, default='concate',
                    help='how to integrate multihead attention')
parser.add_argument('-sample', type=str, default='sample1',
                    help='sample')
parser.add_argument('-cell_type', type=str, default='hHEP',
                    help='cell_type')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class My_Model(nn.Module):
    def __init__(self, encoder_model1):
        super(My_Model, self).__init__()
        self.encoder1 = encoder_model1

    def forward(self, data_feature, adj, train_data):
        index = adj.coalesce().indices()  # 获取adj的indices
        v = adj.coalesce().values()       # 获取index对应的value
        size = adj.coalesce().size()      # 获取adj的行数和列数

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v1 = torch.ones((edge_index1.shape[1])).to(device)
        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        embed1, tf_embed1, target_embed1, pred1 = self.encoder1(data_feature, adj1, train_data)
        embed2, tf_embed2, target_embed2, pred2 = self.encoder1(data_feature, adj2, train_data)

        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2


def gae_train(data_feature, adj1, adj2, gae_model1, optimizer1, scheduler1):

    pretrain_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        gae_model1.train()
        optimizer1.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        z1, train_tf1, train_target1, pred1 = gae_model1(data_feature, adj1, train_x)
        z2, train_tf2, train_target2, pred2 = gae_model1(data_feature, adj2, train_x)

        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss1 = F.binary_cross_entropy(pred1, train_y)
        loss2 = F.binary_cross_entropy(pred2, train_y)

        loss = loss1 + loss2
        loss.backward(retain_graph=True)
        optimizer1.step()
        scheduler1.step()

        pretrain_loss += loss.item()

    return float(pretrain_loss), pred1, pred2


# @torch.no_grad()
def gae_test(test_data, adj1, adj2, gae_model1, gae_model2):
    gae_model1.eval()
    z1 = gae_model1(test_data, adj1)
    index1 = adj1.coalesce().indices()  # 获取adj的indices

    gae_model2.eval()
    z2 = gae_model2(test_data, adj2)
    # index2 = adj2.coalesce().indices()  # 获取adj的indices

    from sklearn.metrics import average_precision_score, roc_auc_score
    neg_edge_index = negative_sampling(index1, z1.size(0))
    pos_y1 = z1.new_ones(index1.size(1))
    neg_y1 = z1.new_zeros(neg_edge_index.size(1))
    y1 = torch.cat([pos_y1, neg_y1], dim=0)
    pos_pred1 = torch.sigmoid((z1[index1[0]] * z1[index1[1]]).sum(dim=1))
    neg_pred1 = torch.sigmoid((z1[neg_edge_index[0]] * z1[neg_edge_index[1]]).sum(dim=1))
    pred1 = torch.cat([pos_pred1, neg_pred1], dim=0)
    y1, pred1 = y1.detach().cpu().numpy(), pred1.detach().cpu().numpy()

    pos_pred2 = torch.sigmoid((z2[index1[0]] * z2[index1[1]]).sum(dim=1))
    neg_pred2 = torch.sigmoid((z2[neg_edge_index[0]] * z2[neg_edge_index[1]]).sum(dim=1))
    pred2 = torch.cat([pos_pred2, neg_pred2], dim=0)
    pred2 = pred2.detach().cpu().numpy()

    return roc_auc_score(y1, pred1), average_precision_score(y1, pred1), roc_auc_score(y1, pred2), average_precision_score(y1, pred2)


def pretrain(data_feature, adj, model1, epochs=10):

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=3e-3)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.99)

    index = adj.coalesce().indices()  # 获取adj的indices
    v = adj.coalesce().values()
    size = adj.coalesce().size()

    for epoch in range(1, epochs + 1):

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v1 = torch.ones((edge_index1.shape[1])).to(device)
        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        pre_train_loss, pred1, pred2 = gae_train(data_feature, adj1, adj2, model1, optimizer1, scheduler1)

        print('Epoch:{}'.format(epoch),
              'pre-train loss:{:.5F}'.format(pre_train_loss))


def train(model, contrast_model, optimizer):

    running_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2 = model(data_feature, adj, train_x)  # 这里实际上不需要tf_embed和target_embed

        con_loss = contrast_model(h1=embed1, h2=embed2)

        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss_BCE1 = F.binary_cross_entropy(pred1, train_y)
        loss_BCE2 = F.binary_cross_entropy(pred2, train_y)
        loss = loss_BCE1 + loss_BCE2 + 0.5 * con_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss


def main_function():
    pre_epochs = 10
    if pre_epochs > 0:
        pretrain(data_feature, adj, encoder_model1, pre_epochs)

    for epoch in range(args.epochs):

        running_loss = train(model, contrast_model, optimizer)
        model.eval()
        _, _, _, _, _, _, _, score = model(data_feature, adj, train_data)
        if args.flag:
            score = torch.softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)

        AUC, AUPR, AUPR_norm = Evaluation(
            y_pred=score, y_true=train_data[:, -1], flag=args.flag)
        print('Epoch:{}'.format(epoch + 1),
              'train loss:{:.5F}'.format(running_loss),
              'AUC:{:.3F}'.format(AUC),
              'AUPR:{:.3F}'.format(AUPR))

    torch.save(model.state_dict(), model_path + '/' + args.cell_type + '1000_i+e_finetune_shared.pkl')


def compute_mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


exp_file = 'Specific Dataset/' + args.cell_type + '/TFs+1000/BL--ExpressionData.csv'
tf_file = 'Specific Dataset/' + args.cell_type + '/TFs+1000/TF.csv'
target_file = 'Specific Dataset/' + args.cell_type + '/TFs+1000/Target.csv'

train_file = 'Data/Specific/' + args.cell_type + ' 1000/' + args.sample + '/Train_set.csv'
test_file = 'Data/Specific/' + args.cell_type + ' 1000/' + args.sample + '/Test_set.csv'
val_file = 'Data/Specific/' + args.cell_type + ' 1000/' + args.sample + '/Validation_set.csv'

data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()       
print('feature shape ', feature.shape)
tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)

# SVD分解降维
# U, S, V = np.linalg.svd(feature.T)
# k = 200
# data_feature = V[:k, :].T
svd_t = TruncatedSVD(n_components=200)
svd_s = TruncatedSVD(n_components=200)
data_feature = svd_t.fit_transform(feature)
print('svd feature shape ', data_feature.shape)

source_feature = pd.read_csv('Specific Dataset/mESC/TFs+1000/BL--ExpressionData.csv', index_col=0)
source_loader = load_data(source_feature)
source_feature = source_loader.exp_data()
source_feature = svd_s.fit_transform(source_feature)
print('source feature shape ', source_feature.shape)

# mmd_before = compute_mmd(data_feature, source_feature)
# data_feature_mean = data_feature.mean(axis=0)
# source_feature_mean = source_feature.mean(axis=0)
# data_feature_align = data_feature - data_feature_mean + source_feature_mean
# mmd_after = compute_mmd(data_feature_align, source_feature)
# data_feature = data_feature_align

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_feature = torch.from_numpy(data_feature)
tf = torch.from_numpy(tf)
data_feature = data_feature.to(device)
tf = tf.to(device)

train_data = pd.read_csv(train_file, index_col=0)
test_data = pd.read_csv(test_file, index_col=0)
validation_data = pd.read_csv(val_file, index_col=0)

data = pd.concat([train_data, validation_data, test_data])
x = data.drop('Label', axis=1)
y = data['Label']
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.95, stratify=y, random_state=42)
train_data = pd.concat([train_data, train_label], axis=1).values
test_data = pd.concat([test_data, test_label], axis=1).values
# train_data = train_data.values
# test_data = test_data.values

train_load = scRNADataset(train_data, data_feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf, loop=args.loop)
adj = adj2saprse_tensor(adj)

train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)

contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)

encoder_model1 = GENELink(input_dim=data_feature.size()[1],
                 hidden1_dim=args.hidden_dim[0],
                 hidden2_dim=args.hidden_dim[1],
                 hidden3_dim=args.hidden_dim[2],
                 output_dim=args.output_dim,
                 num_head1=args.num_head[0],
                 num_head2=args.num_head[1],
                 alpha=args.alpha,
                 device=device,
                 type=args.Type,
                 reduction=args.reduction
                 ).to(device)

adj = adj.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)

model_path = 'model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

model = My_Model(encoder_model1=encoder_model1)
model.load_state_dict(torch.load(model_path + '/source_mESC1000_i+e_False_shared.pkl'))
model = model.to(device)
model.train()

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

# data augmentation methods
# aug1 = A.NodeDropping(pn=0.1)
# aug2 = A.RandomChoice([A.NodeDropping(pn=0.1), A.FeatureMasking(pf=0.1), A.EdgeRemoving(pe=0.1)], 1)
# aug2 = A.NodeDropping(pn=0.1)
# aug2 = A.FeatureMasking(pf=0.1)
aug1 = A.Identity()
aug2 = A.EdgeRemoving(pe=0.3)

main_function()

model.eval()

_, _, _, _, _, _, _, score = model(data_feature, adj, test_data)

if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(
    y_pred=score, y_true=test_data[:, -1], flag=args.flag)

print('test_AUC:{:.3F}'.format(AUC), 'test_AUPR:{:.3F}'.format(AUPR))

np.savetxt('AUC_AUPR_Result/Specific/TF+1000/mESC1000_' + args.sample + '_' + args.cell_type + '_i+e_False_transfer_finetune_nommd.txt', np.array([AUC, AUPR]), fmt='%.3f')



