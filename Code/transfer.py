import pandas as pd
import numpy as np
import random
import os
import sys
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from scGNN import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('-epochs', type=int, default=10, help='Number of epoch.')
parser.add_argument('-num_head', type=list, default=[3, 3], help='Number of head attentions.')
parser.add_argument('-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('-hidden_dim', type=int, default=[128, 64, 32], help='The dimension of hidden layer')
parser.add_argument('-output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('-batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('-loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('-seed', type=int, default=42, help='Random seed')
parser.add_argument('-Type', type=str, default='dot', help='score metric')
parser.add_argument('-flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('-reduction', type=str, default='concate', help='how to integrate multihead attention')
parser.add_argument('-sample', type=str, default='sample1', help='sample')
parser.add_argument('-cell_type', type=str, default='hESC', help='cell_type')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


class GCLink(nn.Module):
    def __init__(self, encoder):
        super(GCLink, self).__init__()
        self.encoder = encoder

    def forward(self, data_feature, adj, train_data):
        index = adj.coalesce().indices()  
        v = adj.coalesce().values()       
        size = adj.coalesce().size()      

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v1 = torch.ones((edge_index1.shape[1])).to(device)
        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        embed1, tf_embed1, target_embed1, pred1 = self.encoder(data_feature, adj1, train_data)
        embed2, tf_embed2, target_embed2, pred2 = self.encoder(data_feature, adj2, train_data)

        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2


def gnn_train(data_feature, adj1, adj2, gnn_model, optimizer, scheduler):

    pretrain_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        gnn_model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        z1, train_tf1, train_target1, pred1 = gnn_model(data_feature, adj1, train_x)
        z2, train_tf2, train_target2, pred2 = gnn_model(data_feature, adj2, train_x)

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
        optimizer.step()
        scheduler.step()

        pretrain_loss += loss.item()

    return float(pretrain_loss), pred1, pred2


def pretrain(data_feature, adj, model, epochs=20):

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    index = adj.coalesce().indices()  
    v = adj.coalesce().values()
    size = adj.coalesce().size()

    for epoch in range(1, epochs + 1):

        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)

        v1 = torch.ones((edge_index1.shape[1])).to(device)
        v2 = torch.ones((edge_index2.shape[1])).to(device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)

        pre_train_loss, pred1, pred2 = gnn_train(data_feature, adj1, adj2, model, optimizer, scheduler)
        print('Epoch:{}'.format(epoch), 'pre-train loss:{:.5F}'.format(pre_train_loss))



def train(model, contrast_model, optimizer):

    running_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        embed1, _, _, pred1, embed2, _, _, pred2 = model(data_feature, adj, train_x)  

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
        pretrain(data_feature, adj, encoder, pre_epochs)

    for epoch in range(args.epochs):

        running_loss = train(model, contrast_model, optimizer)
        print('Epoch:{}'.format(epoch + 1), 'train loss:{:.5F}'.format(running_loss))

    torch.save(model.state_dict(), model_path + '/' + args.cell_type + '_finetuned.pkl')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Data
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

svd_t = TruncatedSVD(n_components=200)
svd_s = TruncatedSVD(n_components=200)
data_feature = svd_t.fit_transform(feature)
print('svd feature shape ', data_feature.shape)

source_feature = pd.read_csv('Specific Dataset/mESC/TFs+1000/BL--ExpressionData.csv', index_col=0)
source_loader = load_data(source_feature)
source_feature = source_loader.exp_data()
source_feature = svd_s.fit_transform(source_feature)
print('source feature shape ', source_feature.shape)

data_feature = torch.from_numpy(data_feature)
tf = torch.from_numpy(tf)

data_feature = data_feature.to(device)
tf = tf.to(device)

train_data = pd.read_csv(train_file, index_col=0)
test_data = pd.read_csv(test_file, index_col=0)
validation_data = pd.read_csv(val_file, index_col=0)

# Only 5% of the data is used for fine-tuning, and the rest is used as a test set.
data = pd.concat([train_data, validation_data, test_data])
x = data.drop('Label', axis=1)
y = data['Label']
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.95, stratify=y, random_state=42)
train_data = pd.concat([train_data, train_label], axis=1).values
test_data = pd.concat([test_data, test_label], axis=1).values

train_load = scRNADataset(train_data, data_feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf, loop=args.loop)
adj = adj2saprse_tensor(adj)

train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)

# Construct Model
contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)
encoder = GENELink(input_dim=data_feature.size()[1],
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

model = GCLink(encoder=encoder)
model.load_state_dict(torch.load(model_path + '/source_mESC1000.pkl'))
model = model.to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

# data augmentation 
aug1 = A.Identity()
aug2 = A.EdgeRemoving(pe=0.3)

model.train()
main_function()

model.eval()

_, _, _, _, _, _, _, score = model(data_feature, adj, test_data)

if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=args.flag)

print('test_AUC:{:.3F}'.format(AUC), 'test_AUPR:{:.3F}'.format(AUPR))
