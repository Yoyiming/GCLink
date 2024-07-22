import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
from torch_geometric.nn import GAE, GCNConv, VGAE, GATConv
import GCL.augmentors as A

class GCN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, out_channels, cached=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_temp1 = self.conv1(x, edge_index).tanh()
        x_temp2 = self.dropout(x_temp1)
        return self.conv2(x_temp2, edge_index)


class GENELink(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_head1, num_head2,
                 alpha, device, type, reduction):
        super(GENELink, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim

        self.ConvLayer1 = [AttentionLayer(
            input_dim, hidden1_dim, alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)

        self.ConvLayer2 = [AttentionLayer(
            self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i), attention)

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)

        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

    # 将基因表达数据和邻接矩阵输入，得到图
    def encode(self, x, adj):

        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
            x = F.elu(x)
        elif self.reduction == 'mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)
        else:
            raise TypeError

        out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]), dim=0)
        return out

    def decode(self, tf_embed, target_embed):

        if self.type == 'dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob, dim=1).view(-1, 1)
            return prob
        elif self.type == 'cosine':
            prob = torch.cosine_similarity(
                tf_embed, target_embed, dim=1).view(-1, 1)
            return prob
        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))

    def forward(self, x, adj, train_sample):

        # GCN代替GAT
        # NUM_FEATURES1 = x.shape[1]
        # encoder_mode = GAE(GCN_Encoder(NUM_FEATURES1, 128, 64, 0.6))
        # encoder_mode = encoder_mode.to(self.device)
        # index = adj.coalesce().indices()
        # embed = encoder_mode.encode(x, index)

        embed = self.encode(x, adj)
        # adj = torch.matmul(embed, embed.t())
        # adj = torch.sigmoid(adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)


        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]
        pred = self.decode(train_tf, train_target)

        return embed, tf_embed, target_embed, pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        # torch.nn.Parameter可以将tensor变成可训练的， torch.FloatTensor将list ,numpy转化为tensor
        self.weight = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim, 1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # 初始化权重和bias
    def reset_parameters(self):
        # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，这里有一个gain，增益的大小是依据激活函数类型来设定
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T, negative_slope=self.alpha)
        return e

    def forward(self, x, adj):

        h = torch.matmul(x, self.weight)  # h = XW
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
        # 如果满足条件，则选择e，否则选择zero_vec作为输出
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.softmax(e, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass

        output_data = F.leaky_relu(output_data, negative_slope=self.alpha)
        output_data = F.normalize(output_data, p=2, dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data
