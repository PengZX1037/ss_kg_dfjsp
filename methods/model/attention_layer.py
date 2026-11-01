import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleOpAttnBlock(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_prob):
        super(SingleOpAttnBlock, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, op_mask):
        projected_features = torch.matmul(h, self.W)
        sz_b, N, _ = projected_features.size()
        neighbor_features = torch.stack([projected_features.roll(1, dims=1), projected_features, projected_features.roll(-1, dims=1)], dim=-2)
        source_attention_scores = torch.matmul(projected_features, self.a[:self.out_features, :])
        neighbor_attention_scores = torch.matmul(projected_features, self.a[self.out_features:, :])
        shifted_neighbor_scores = torch.stack([neighbor_attention_scores.roll(1, dims=1), neighbor_attention_scores, neighbor_attention_scores.roll(-1, dims=1)], dim=-1)
        e = source_attention_scores.unsqueeze(-1) + shifted_neighbor_scores
        e = self.leaky_relu(e)
        masked_attention_scores = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(op_mask.unsqueeze(-2) > 0, masked_attention_scores, e)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        updated_features = torch.matmul(attention, neighbor_features).squeeze(-2)
        return updated_features

class MultiHeadOpAttnBlock(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        super(MultiHeadOpAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.attentions = [SingleOpAttnBlock(input_dim, output_dim, dropout_prob) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, op_mask):
        h = self.dropout(h)
        head_outputs = [att(h, op_mask) for att in self.attentions]
        if self.concat:
            h = torch.cat(head_outputs, dim=-1)
        else:
            h = torch.stack(head_outputs, dim=-1)
            h = h.mean(dim=-1)
        return h if self.activation is None else self.activation(h)

class SingleMchAttnBlock(nn.Module):

    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob):
        super(SingleMchAttnBlock, self).__init__()
        self.node_in_features = node_input_dim
        self.edge_in_features = edge_input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.W = nn.Parameter(torch.empty(size=(node_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W_edge = nn.Parameter(torch.empty(size=(edge_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, mch_mask, comp_val):
        projected_nodes = torch.matmul(h, self.W)
        projected_edges = torch.matmul(comp_val, self.W_edge)
        e = self.get_attention_coef(projected_nodes, projected_edges)
        masked_attention_scores = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(mch_mask > 0, e, masked_attention_scores)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        updated_features = torch.matmul(attention, projected_nodes)
        return updated_features

    def get_attention_coef(self, Wh, W_edge):
        source_attention_scores = torch.matmul(Wh, self.a[:self.out_features, :])
        target_attention_scores = torch.matmul(Wh, self.a[self.out_features:2 * self.out_features, :])
        edge_attention_scores = torch.matmul(W_edge, self.a[2 * self.out_features:, :])
        e = source_attention_scores + target_attention_scores.transpose(-1, -2) + edge_attention_scores.squeeze(-1)
        return self.leaky_relu(e)

class MultiHeadMchAttnBlock(nn.Module):

    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        super(MultiHeadMchAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.concat = concat
        self.activation = activation
        self.num_heads = num_heads
        self.attentions = [SingleMchAttnBlock(node_input_dim, edge_input_dim, output_dim, dropout_prob) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, mch_mask, comp_val):
        h = self.dropout(h)
        head_outputs = [att(h, mch_mask, comp_val) for att in self.attentions]
        if self.concat:
            h = torch.cat(head_outputs, dim=-1)
        else:
            h = torch.stack(head_outputs, dim=-1)
            h = h.mean(dim=-1)
        return h if self.activation is None else self.activation(h)
