import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb
from args import *
import numpy as np
####################################################################
args = make_args()
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

torch.manual_seed(2020) # seed for reproducible numbers




class GAT_Attention(nn.Module):
    def __init__(self,orignal_features, anchor_features, hidden_features, dropout=0.5, alpha=0.2, nheads=1):
        super(GAT_Attention, self).__init__()
        self.dropout = dropout

        self.attentions = [GATLayer(orignal_features,anchor_features, hidden_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

    def forward(self,orignal_x, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(orignal_x,x, adj) for att in self.attentions], dim=1)
        #x = torch.sum(torch.stack([att(x, edge_list) for att in self.attentions]), dim=0) / len(self.attentions)
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class GATLayer(nn.Module):
    def __init__(self,orignal_features, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout             = dropout       
        self.orignal_features    = orignal_features 
        self.in_features         = in_features    # 
        self.out_features        = out_features   # 
        self.alpha               = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat              = concat         # Always Set to True
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,orignal_input, input, adj):

        h = torch.matmul(orignal_input, self.W) #orignal_input(N*feature_size)
        N = h.size()[0] # Num of nodes
        f = h.size()[1] #features
        k = adj.size()[1] # Num of anchors

        h_i=h.repeat(1, k).view(N,k, f)

        input_reshape=input.view(N*k,f)
        h_j = torch.matmul(input_reshape, self.W)
        h_j=h_j.view(N,k,f)
        # Attention Mechanism
        a_input = torch.cat([h_i,h_j],dim=2)

        e       = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        atten=attention.reshape(N,k,1)
        

        h_prime   = atten* h_j
        
        h_prime = torch.sum(h_prime, dim=1)  # n*d
        if args.attentionAddSelf:
            h_prime = (h_prime+h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

###############################################################
class GraphReach_Layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(GraphReach_Layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)
        
        self.linear_hidden = nn.Linear(input_dim*2, output_dim)

        if args.attention:
            self.linear_hidden_orignal_features = nn.Linear(input_dim, output_dim)
            self.gat_layer = GAT_Attention(output_dim,output_dim, output_dim)#,0.5,0.2)

        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

        

    def forward(self, feature, dists_max, dists_argmax,dists_max2):
        if self.dist_trainable:
            
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()
            dists_max2 = self.dist_compute(dists_max2.unsqueeze(-1)).squeeze()
            
        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)        
        
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)* dists_max2.unsqueeze(-1)
        
        messages = torch.cat((messages, self_feature), dim=-1)
        

        messages = self.linear_hidden(messages).squeeze()
        
        messages = self.act(messages) # n*m*d
        
        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out embedding
        
        if args.attention:
            feature=self.linear_hidden_orignal_features(feature)
            out_structure = self.gat_layer(feature,messages,dists_max).squeeze()
        else:
            
            out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class GraphReach(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=1, dropout=True, **kwargs):
        super(GraphReach, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = GraphReach_Layer(feature_dim, hidden_dim)
        else:
            self.conv_first = GraphReach_Layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([GraphReach_Layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = GraphReach_Layer(hidden_dim, output_dim)

        self.lin2 = nn.Linear(final_emb_size, num_class)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)

        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax,data.dists_max2)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)

        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax,data.dists_max2)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax,data.dists_max2)
        x_position = F.normalize(x_position, p=2, dim=-1)
        x_position = F.dropout(x_position, training=self.training)
        x_position = self.lin2(x_position)
        return F.log_softmax(x_position, dim=1)

####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)

        self.lin2 = nn.Linear(output_dim, num_class)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        ######
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)


        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
        

class SAGE(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GIN(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


