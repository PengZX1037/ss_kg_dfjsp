from common_utils import nonzero_averaging
from model.attention_layer import *
from model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from params import configs

class DualAttentionNetwork(nn.Module):
    """Encode operation and machine features with dual attention."""

    def __init__(self, config):
        super(DualAttentionNetwork, self).__init__()
        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()
        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob
        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB
        mid_dim = self.output_dim_per_layer[:-1]
        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim
        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim
        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()
        for i in range(self.num_dan_layers):
            current_j_input_dim = num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i]
            self.op_attention_blocks.append(MultiHeadOpAttnBlock(input_dim=current_j_input_dim, num_heads=self.num_heads_OAB[i], output_dim=self.output_dim_per_layer[i], concat=True if i < self.num_dan_layers - 1 else False, activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate, dropout_prob=self.dropout_prob))
        for i in range(self.num_dan_layers):
            current_m_node_input_dim = num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i]
            current_m_edge_input_dim = num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i]
            self.mch_attention_blocks.append(MultiHeadMchAttnBlock(node_input_dim=current_m_node_input_dim, edge_input_dim=current_m_edge_input_dim, num_heads=self.num_heads_MAB[i], output_dim=self.output_dim_per_layer[i], concat=True if i < self.num_dan_layers - 1 else False, activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate, dropout_prob=self.dropout_prob))

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        sz_b, M, _, J = comp_idx.size()
        N = fea_j.size(1)
        op_mask_float = op_mask.float()
        mch_mask_float = mch_mask.float()
        flattened_competition_indices = comp_idx.reshape(sz_b, -1, J)
        operation_states = fea_j
        machine_states = fea_m
        for layer in range(self.num_dan_layers):
            current_op_feature_dim = operation_states.size(-1)
            zero_candidate_states = torch.zeros(sz_b, J, current_op_feature_dim, device=operation_states.device, dtype=operation_states.dtype)
            candidate_is_valid = candidate >= 0
            expanded_candidate_mask = candidate_is_valid.unsqueeze(-1).expand(-1, -1, current_op_feature_dim)
            safe_candidate_indices = candidate.long().clamp(min=0, max=N - 1)
            candidate_gather_indices = safe_candidate_indices.unsqueeze(-1).expand(-1, -1, current_op_feature_dim)
            candidate_states = torch.gather(operation_states, 1, candidate_gather_indices)
            candidate_job_features = torch.where(expanded_candidate_mask, candidate_states, zero_candidate_states)
            competition_features = torch.matmul(flattened_competition_indices.float(), candidate_job_features.float()).reshape(sz_b, M, M, -1)
            operation_states = self.op_attention_blocks[layer](operation_states, op_mask_float)
            machine_states = self.mch_attention_blocks[layer](machine_states, mch_mask_float, competition_features)
        fea_j_out = operation_states
        fea_m_out = machine_states
        fea_j_global = nonzero_averaging(fea_j_out)
        fea_m_global = nonzero_averaging(fea_m_out)
        return (fea_j_out, fea_m_out, fea_j_global, fea_m_global)

class DANIEL(nn.Module):
    """Produce dispatch/WAIT probabilities and a state-value estimate."""

    def __init__(self, config):
        super(DANIEL, self).__init__()
        device = torch.device(config.device)
        self.pair_input_dim = config.pair_input_dim
        self.embedding_output_dim = config.layer_fea_output_dim[-1]
        self.actor_input_dim = config.actor_input_dim
        self.critic_input_dim = config.critic_input_dim
        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(config.num_mlp_layers_actor, self.actor_input_dim, config.hidden_dim_actor, 1).to(device)
        self.actor_noop_head = Actor(config.num_mlp_layers_critic, self.critic_input_dim, config.hidden_dim_critic, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, self.critic_input_dim, config.hidden_dim_critic, 1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs, wait_action_valid=None):
        fea_j_out, fea_m_out, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask.float(), candidate, fea_m, mch_mask.float(), comp_idx)
        sz_b, J = candidate.size()
        M = fea_m_out.size(1)
        embedding_dim = fea_j_out.size(-1)
        valid_candidate_mask = candidate >= 0
        valid_candidate_mask_expanded_j = valid_candidate_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)
        default_fea_j = torch.zeros(sz_b, J, embedding_dim, device=fea_j_out.device, dtype=fea_j_out.dtype)
        valid_candidate_indices = candidate.long().clamp(min=0)
        candidate_idx = valid_candidate_indices.unsqueeze(-1).expand(-1, -1, embedding_dim)
        gathered_fea_j = torch.gather(fea_j_out, 1, candidate_idx)
        candidate_operation_features = torch.where(valid_candidate_mask_expanded_j, gathered_fea_j, default_fea_j)
        tiled_operation_features = candidate_operation_features.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, J * M, embedding_dim)
        tiled_machine_features = fea_m_out.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, J * M, embedding_dim)
        global_operation_features = fea_j_global.unsqueeze(1).expand_as(tiled_operation_features)
        global_machine_features = fea_m_global.unsqueeze(1).expand_as(tiled_operation_features)
        flattened_pair_features = fea_pairs.reshape(sz_b, J * M, self.pair_input_dim)
        actor_features = torch.cat((tiled_operation_features, tiled_machine_features, global_operation_features, global_machine_features, flattened_pair_features), dim=-1)
        if actor_features.size(-1) != self.actor_input_dim:
            raise ValueError(f'Actor input dimension mismatch. Expected {self.actor_input_dim}, got {actor_features.size(-1)}')
        dispatch_logits = self.actor(actor_features)
        dispatch_logits = dispatch_logits.squeeze(-1)
        critic_features = torch.cat((fea_j_global, fea_m_global), dim=-1)
        wait_logit = self.actor_noop_head(critic_features)
        invalid_dispatch_mask = dynamic_pair_mask.reshape(sz_b, -1).bool().to(dispatch_logits.device)
        if wait_action_valid is None:
            wait_action_valid = torch.ones(sz_b, dtype=torch.bool, device=dispatch_logits.device)
        else:
            wait_action_valid = torch.as_tensor(wait_action_valid, device=dispatch_logits.device, dtype=torch.bool).reshape(-1)
            if wait_action_valid.numel() != sz_b:
                raise ValueError(f'wait_action_valid length {wait_action_valid.numel()} != batch {sz_b}')
        invalid_action_mask = torch.cat([invalid_dispatch_mask, (~wait_action_valid).unsqueeze(1)], dim=1)
        if torch.any(invalid_action_mask.all(dim=1)):
            invalid_rows = torch.where(invalid_action_mask.all(dim=1))[0].detach().cpu().tolist()
            raise RuntimeError(f'No feasible dispatch OR advancing WAIT action in batch rows {invalid_rows}; check window W and environment time updates')
        action_logits = torch.cat((dispatch_logits, wait_logit), dim=1)
        action_probabilities = F.softmax(action_logits.masked_fill(invalid_action_mask, float('-inf')), dim=1)
        if not torch.isfinite(action_probabilities).all():
            raise RuntimeError('Non-finite actor probabilities after feasibility masking')
        if critic_features.size(-1) != self.critic_input_dim:
            raise ValueError(f'Critic input dimension mismatch. Expected {self.critic_input_dim}, got {critic_features.size(-1)}')
        state_value = self.critic(critic_features)
        return (action_probabilities, state_value)
