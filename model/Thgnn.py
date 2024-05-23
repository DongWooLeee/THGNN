from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math

class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        #MULTIHEAD ATTENTION을 위한 것 -> weight: gru를 통과하므로 [hidden_dim, num_heads*out_features]
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features)) # in_features -> num_heads*out_features
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        #Normalizing Term in Attention Weight Calculation
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight) # inputs: (num_nodes, in_features), support: (num_nodes, num_heads*out_features)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))# support: (num_heads, num_nodes, out_features)
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1) # f_1: (num_heads, 1, num_nodes)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1) # f_2: (num_heads, num_nodes, 1)
        logits = f_1 + f_2 # logits: (num_heads, num_nodes, num_nodes)
        weight = self.leaky_relu(logits) # weight: (num_heads, num_nodes, num_nodes)
        print('weight 텐서의 shape:',weight.shape) 
        print('---------------')
        print('adj_mat 텐서의 shape:',adj_mat.shape) # adj_mat: (num_nodes_end, num_nodes_end) -> 근데 이건... start~ end 기준으로 다른 adj_mat을 넣어야 계산이 가능함.       
        # 만약 이걸 그대로 쓰려면, predict하는 시점 기준으로 과거 시점에서 가져올 노드 개수가 모두 같아야함-> 맞춰줄 수 있음.
        masked_weight = torch.mul(weight, adj_mat).to_sparse() # 
        attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):    
    '''    
    Ways to prevent oversmoothing in GNNs.
    Source: https://github.com/LingxiaoShawn/PairNorm/blob/master/layers.py

    '''
    def __init__(self, mode='PN', scale=1): # Pair-Normalizaiton, initializing scale to 1
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0) 
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class GraphAttnSemIndividual(Module):
    '''
    Node 별로 각 관계에 대한 attention을 구하는 방식. Positive와 negative, 그리고 self로부터 온 hidden state에 대한 attention을 구하게 된다. 
    '''
    
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(GraphAttnSemIndividual, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False)) # 같은 차원으로서 보내주기 위함임/
    '''
    Usage:
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,..)
        support = support.squeeze()
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)
        all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
    
    '''
    ## Beta는 HeteroGeneous Graph Attention Network의 산출물임.
    # MLP 기반 ATTENTION
    def forward(self, inputs, requires_weight=False):
        w = self.project(inputs) 
        beta = torch.softmax(w, dim=1) # Beta를 구하기
        if requires_weight:
            return (beta * inputs).sum(1), beta
        else:
            return (beta * inputs).sum(1), None


class StockHeteGAT(nn.Module):
    def __init__(self, in_features=5, out_features=8, num_heads=8, hidden_dim=64, num_layers=1):
        super(StockHeteGAT, self).__init__()
        self.encoding = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(out_features*num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(out_features*num_heads, hidden_dim)
        
        self.pn = PairNorm(mode='PN-SI')
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear): # Linear layer에 대해서는 xavier_uniform으로 초기화
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, pos_adj, neg_adj, requires_weight=False):
        _, support = self.encoding(inputs)
        
        '''
        GRU 의 output 형태는 다음과 같다.
        
        
        output: 크기가 (batch_size, sequence_length, hidden_size)인 텐서. 이 출력은 각 시간 스텝의 은닉 상태를 포함하고 있습니다. 이 출력은 사용되지 않습니다.
        support: 크기가 (batch_size, hidden_size)인 텐서. 이것은 GRU의 마지막 은닉 상태를 나타냅니다. 이 출력은 입력 시퀀스에 대한 요약 정보로 간주될 수 있으며, 이후 다른 레이어에 입력으로 사용됩니다.
        
        '''
        support = support.squeeze() # support : (batch_size, hidden_size) => batch_size는 여기서 노드의 개수를 의미함.
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight) # pos_support: (batch_size, hidden_size), pos_attn_weights: (batch_size, num_heads, num_nodes)
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)
        all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
        all_embedding = self.pn(all_embedding)
        if requires_weight:
            return self.predictor(all_embedding), (pos_attn_weights, neg_attn_weights, sem_attn_weights)
        else:
            return self.predictor(all_embedding)
