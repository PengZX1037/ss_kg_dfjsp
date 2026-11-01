from model.main_model import *
from common_utils import eval_actions
import torch.nn as nn
import torch
from copy import deepcopy
from params import configs
import numpy as np
import warnings

class Memory:
    """Store sampled trajectories and prepare batches for PPO."""

    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = []
        self.action_seq = []
        self.reward_seq = []
        self.val_seq = []
        self.done_seq = []
        self.log_probs = []

    def clear_memory(self):
        del self.states[:]
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]
        if hasattr(self, 't_old_val_seq'):
            self.t_old_val_seq = None

    def push(self, state):
        self.states.append(state)

    def transpose_and_flatten_data(self):
        if not self.states:
            return None
        operation_features = [s.fea_j_tensor for s in self.states]
        operation_masks = [s.op_mask_tensor for s in self.states]
        machine_features = [s.fea_m_tensor for s in self.states]
        machine_masks = [s.mch_mask_tensor for s in self.states]
        dispatch_masks = [s.dynamic_pair_mask_tensor for s in self.states]
        competition_indices = [s.comp_idx_tensor for s in self.states]
        candidate_indices = [s.candidate_tensor for s in self.states]
        pair_features = [s.fea_pairs_tensor for s in self.states]
        wait_action_masks = [s.wait_action_valid_tensor for s in self.states]
        flat_operation_features = torch.stack(operation_features, dim=0).transpose(0, 1).flatten(0, 1)
        flat_operation_masks = torch.stack(operation_masks, dim=0).transpose(0, 1).flatten(0, 1)
        flat_machine_features = torch.stack(machine_features, dim=0).transpose(0, 1).flatten(0, 1)
        flat_machine_masks = torch.stack(machine_masks, dim=0).transpose(0, 1).flatten(0, 1)
        flat_dispatch_masks = torch.stack(dispatch_masks, dim=0).transpose(0, 1).flatten(0, 1)
        flat_competition_indices = torch.stack(competition_indices, dim=0).transpose(0, 1).flatten(0, 1)
        flat_candidate_indices = torch.stack(candidate_indices, dim=0).transpose(0, 1).flatten(0, 1)
        flat_pair_features = torch.stack(pair_features, dim=0).transpose(0, 1).flatten(0, 1)
        flat_wait_action_masks = torch.stack(wait_action_masks, dim=0).transpose(0, 1).flatten(0, 1)
        flat_actions = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        flat_rewards = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        flat_values = self.t_old_val_seq.flatten(0, 1)
        flat_done_flags = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        flat_log_probs = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)
        return (flat_operation_features, flat_operation_masks, flat_machine_features, flat_machine_masks, flat_dispatch_masks,
                flat_competition_indices, flat_candidate_indices, flat_pair_features, flat_actions, flat_rewards,
                flat_values, flat_done_flags, flat_log_probs, flat_wait_action_masks)

    def get_gae_advantages(self):
        if not hasattr(self, 't_old_val_seq') or self.t_old_val_seq is None:
            raise RuntimeError('Call transpose_and_flatten_data before get_gae_advantages.')
        if not self.reward_seq:
            return (torch.tensor([]), torch.tensor([]))
        reward_values = torch.stack(self.reward_seq, dim=0)
        state_values = self.t_old_val_seq.transpose(0, 1)
        terminal_flags = torch.stack(self.done_seq, dim=0).float()
        trajectory_length, environment_count = reward_values.shape
        advantage = torch.zeros(environment_count, device=state_values.device)
        advantage_history = []
        for i in reversed(range(trajectory_length)):
            next_state_value = state_values[i + 1] if i < trajectory_length - 1 else torch.zeros_like(state_values[i])
            next_value_adjusted = next_state_value * (1.0 - terminal_flags[i])
            temporal_difference = reward_values[i] + self.gamma * next_value_adjusted - state_values[i]
            advantage = temporal_difference + self.gamma * self.gae_lambda * advantage * (1.0 - terminal_flags[i])
            advantage_history.insert(0, advantage)
        batched_advantages = torch.stack(advantage_history, dim=0).transpose(0, 1).to(torch.float32)
        value_targets = (batched_advantages + self.t_old_val_seq).flatten(0, 1)
        flattened_advantages = batched_advantages.flatten(0, 1)
        normalized_advantages = (flattened_advantages - flattened_advantages.mean()) / (flattened_advantages.std() + 1e-08)
        return (normalized_advantages, value_targets)

class PPO:
    """Optimize the scheduling policy using PPO or guided imitation."""

    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau
        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size
        self.policy = DANIEL(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()

    def update(self, memory):
        trajectory_batch = memory.transpose_and_flatten_data()
        if trajectory_batch is None:
            warnings.warn('PPO update skipped: Memory is empty.')
            return (0.0, 0.0)
        (flat_operation_features, flat_operation_masks, flat_machine_features, flat_machine_masks, flat_dispatch_masks,
         flat_competition_indices, flat_candidate_indices, flat_pair_features, flat_actions, flat_rewards, flat_values,
         flat_done_flags, flat_log_probs, flat_wait_action_masks) = trajectory_batch
        if flat_actions.numel() == 0:
            warnings.warn('PPO update skipped: No actions found in memory.')
            return (0.0, 0.0)
        try:
            t_advantage_seq, v_target_seq = memory.get_gae_advantages()
            expected_len = flat_actions.shape[0]
            if t_advantage_seq.shape[0] != expected_len or v_target_seq.shape[0] != expected_len:
                raise ValueError(f'Shape mismatch after GAE: Adv {t_advantage_seq.shape}, V_target {v_target_seq.shape}, Expected {expected_len}')
        except Exception as e:
            warnings.warn(f'Error calculating GAE: {e}')
            return (0.0, 0.0)
        full_batch_size = flat_actions.shape[0]
        current_minibatch_size = min(self.minibatch_size, full_batch_size)
        if current_minibatch_size <= 0:
            warnings.warn(f'PPO update skipped: No data available for minibatching (full_batch_size={full_batch_size}).')
            return (0.0, 0.0)
        num_minibatches = full_batch_size // current_minibatch_size
        loss_epochs = 0
        v_loss_epochs = 0
        indices = np.arange(full_batch_size)
        for k_epoch in range(self.k_epochs):
            np.random.shuffle(indices)
            for i in range(num_minibatches):
                start_idx = i * current_minibatch_size
                end_idx = (i + 1) * current_minibatch_size
                minibatch_indices = indices[start_idx:end_idx]
                mb_actions = flat_actions[minibatch_indices]
                valid_transition_mask = mb_actions >= 0
                selected_transition_indices = minibatch_indices[valid_transition_mask.cpu().numpy()]
                if len(selected_transition_indices) == 0:
                    continue
                valid_fea_j = flat_operation_features[selected_transition_indices]
                valid_op_mask = flat_operation_masks[selected_transition_indices]
                valid_candidate = flat_candidate_indices[selected_transition_indices]
                valid_fea_m = flat_machine_features[selected_transition_indices]
                valid_mch_mask = flat_machine_masks[selected_transition_indices]
                valid_comp_idx = flat_competition_indices[selected_transition_indices]
                valid_dynamic_mask = flat_dispatch_masks[selected_transition_indices]
                valid_fea_pairs = flat_pair_features[selected_transition_indices]
                valid_wait_action = flat_wait_action_masks[selected_transition_indices]
                valid_actions = flat_actions[selected_transition_indices]
                valid_old_logprobs = flat_log_probs[selected_transition_indices]
                valid_advantages = t_advantage_seq[selected_transition_indices]
                valid_v_targets = v_target_seq[selected_transition_indices]
                try:
                    pis, vals = self.policy(fea_j=valid_fea_j, op_mask=valid_op_mask, candidate=valid_candidate, fea_m=valid_fea_m,
                                            mch_mask=valid_mch_mask, comp_idx=valid_comp_idx, dynamic_pair_mask=valid_dynamic_mask,
                                            fea_pairs=valid_fea_pairs, wait_action_valid=valid_wait_action)
                except Exception as e:
                    warnings.warn(f'\nError during policy forward pass in PPO update (epoch {k_epoch}, mb {i}): {e}')
                    continue
                try:
                    logprobs, ent_loss = eval_actions(pis, valid_actions)
                except ValueError as e:
                    warnings.warn(f'\nError during eval_actions in PPO update (epoch {k_epoch}, mb {i}): {e}')
                    action_dim = pis.shape[-1]
                    invalid_action_mask = (valid_actions < 0) | (valid_actions >= action_dim)
                    if torch.any(invalid_action_mask):
                        pass
                    continue
                if logprobs.shape != valid_old_logprobs.shape:
                    warnings.warn(f'Logprob shape mismatch: new {logprobs.shape}, old {valid_old_logprobs.shape}')
                    continue
                probability_ratios = torch.exp(logprobs - valid_old_logprobs.detach())
                if probability_ratios.shape != valid_advantages.shape:
                    warnings.warn(f'Ratio/Advantage shape mismatch: ratio {probability_ratios.shape}, adv {valid_advantages.shape}')
                    continue
                unclipped_objective = probability_ratios * valid_advantages
                clipped_objective = torch.clamp(probability_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * valid_advantages
                if vals.squeeze(1).shape != valid_v_targets.shape:
                    warnings.warn(f'Value/Target shape mismatch: val {vals.squeeze(1).shape}, target {valid_v_targets.shape}')
                    continue
                value_loss = self.V_loss_2(vals.squeeze(1), valid_v_targets)
                policy_loss = -torch.min(unclipped_objective, clipped_objective)
                ent_loss = -ent_loss
                loss = self.vloss_coef * value_loss + self.ploss_coef * policy_loss.mean() + self.entloss_coef * ent_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epochs += loss.item()
                v_loss_epochs += value_loss.item()
        if self.tau > 0:
            for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
                policy_old_params.data.copy_(self.tau * policy_params.data + (1 - self.tau) * policy_old_params.data)
        else:
            self.policy_old.load_state_dict(self.policy.state_dict())
        mean_total_loss = loss_epochs / (self.k_epochs * num_minibatches) if num_minibatches > 0 else 0
        mean_value_loss = v_loss_epochs / (self.k_epochs * num_minibatches) if num_minibatches > 0 else 0
        return (mean_total_loss, mean_value_loss)

    def update_knowledge_guided(self, states_list, target_actions_list):
        if not states_list or not target_actions_list:
            warnings.warn('KGRL update skipped: Empty states or actions list.')
            return 0.0
        device = self.device
        max_operation_count = 0
        max_job_count = 0
        op_feat_dim = states_list[0].fea_j_tensor.shape[-1]
        mch_feat_dim = states_list[0].fea_m_tensor.shape[-1]
        pair_feat_dim = states_list[0].fea_pairs_tensor.shape[-1]
        num_machines = states_list[0].fea_m_tensor.shape[-2]
        for state in states_list:
            if len(state.fea_j_tensor.shape) >= 3:
                max_operation_count = max(max_operation_count, state.fea_j_tensor.shape[1])
            if len(state.candidate_tensor.shape) >= 2:
                max_job_count = max(max_job_count, state.candidate_tensor.shape[1])
        if max_operation_count == 0 or max_job_count == 0:
            warnings.warn('KGRL update skipped: Could not determine max dimensions from states.')
            return 0.0
        padded_operation_features = []
        padded_op_mask_list = []
        padded_fea_m_list = []
        padded_mch_mask_list = []
        padded_dynamic_mask_list = []
        padded_comp_idx_list = []
        padded_candidate_list = []
        padded_fea_pairs_list = []
        wait_valid_list = []
        for state in states_list:
            state_device = state.fea_j_tensor.device
            if state_device != device:
                warnings.warn(f'State tensor found on {state_device}, expected {device}. Moving.')
            s_fea_j = state.fea_j_tensor.to(device)
            operation_padding = max_operation_count - s_fea_j.shape[1]
            if operation_padding < 0:
                raise ValueError('Padding calculation error (N)')
            padded_fea_j = F.pad(s_fea_j, (0, 0, 0, operation_padding), mode='constant', value=0)
            padded_operation_features.append(padded_fea_j)
            s_op_mask = state.op_mask_tensor.to(device)
            operation_padding = max_operation_count - s_op_mask.shape[1]
            if operation_padding < 0:
                raise ValueError('Padding calculation error (op_mask N)')
            padded_op_mask = F.pad(s_op_mask, (0, 0, 0, operation_padding), mode='constant', value=1)
            padded_op_mask_list.append(padded_op_mask)
            padded_fea_m_list.append(state.fea_m_tensor.to(device))
            padded_mch_mask_list.append(state.mch_mask_tensor.to(device))
            s_dyn_mask = state.dynamic_pair_mask_tensor.to(device)
            job_padding = max_job_count - s_dyn_mask.shape[1]
            if job_padding < 0:
                raise ValueError('Padding calculation error (dyn_mask J)')
            padded_dyn_mask = F.pad(s_dyn_mask, (0, 0, 0, job_padding), mode='constant', value=True)
            padded_dynamic_mask_list.append(padded_dyn_mask)
            s_comp_idx = state.comp_idx_tensor.to(device)
            job_padding = max_job_count - s_comp_idx.shape[3]
            if job_padding < 0:
                raise ValueError('Padding calculation error (comp_idx J)')
            padded_comp_idx = F.pad(s_comp_idx, (0, job_padding), mode='constant', value=0)
            padded_comp_idx_list.append(padded_comp_idx)
            s_candidate = state.candidate_tensor.to(device)
            job_padding = max_job_count - s_candidate.shape[1]
            if job_padding < 0:
                raise ValueError('Padding calculation error (candidate J)')
            padded_candidate = F.pad(s_candidate, (0, job_padding), mode='constant', value=-1)
            padded_candidate_list.append(padded_candidate)
            s_fea_pairs = state.fea_pairs_tensor.to(device)
            job_padding = max_job_count - s_fea_pairs.shape[1]
            if job_padding < 0:
                raise ValueError('Padding calculation error (fea_pairs J)')
            padded_fea_pairs = F.pad(s_fea_pairs, (0, 0, 0, 0, 0, job_padding), mode='constant', value=0)
            padded_fea_pairs_list.append(padded_fea_pairs)
            if not hasattr(state, 'wait_action_valid_tensor'):
                raise RuntimeError('KG replay state lacks the WAIT feasibility mask')
            wait_valid_list.append(state.wait_action_valid_tensor.to(device).reshape(-1))
        try:
            fea_j_batch = torch.cat(padded_operation_features, dim=0)
            op_mask_batch = torch.cat(padded_op_mask_list, dim=0)
            fea_m_batch = torch.cat(padded_fea_m_list, dim=0)
            mch_mask_batch = torch.cat(padded_mch_mask_list, dim=0)
            dynamic_mask_batch = torch.cat(padded_dynamic_mask_list, dim=0)
            comp_idx_batch = torch.cat(padded_comp_idx_list, dim=0)
            candidate_batch = torch.cat(padded_candidate_list, dim=0)
            fea_pairs_batch = torch.cat(padded_fea_pairs_list, dim=0)
            translated_targets = []
            for st, target in zip(states_list, target_actions_list):
                action_id = int(torch.as_tensor(target).item())
                local_wait_id = int(st.candidate_tensor.shape[1]) * num_machines
                if action_id == local_wait_id:
                    action_id = max_job_count * num_machines
                translated_targets.append(action_id)
            target_action_tensor = torch.as_tensor(translated_targets, device=device, dtype=torch.long)
            wait_valid_batch = torch.cat(wait_valid_list, dim=0)
        except Exception as e:
            warnings.warn(f'Error stacking padded tensors for KGRL update: {e}')
            return 0.0
        try:
            pis, _ = self.policy(fea_j=fea_j_batch, op_mask=op_mask_batch, candidate=candidate_batch, fea_m=fea_m_batch,
                                 mch_mask=mch_mask_batch, comp_idx=comp_idx_batch, dynamic_pair_mask=dynamic_mask_batch,
                                 fea_pairs=fea_pairs_batch, wait_action_valid=wait_valid_batch)
        except Exception as e:
            warnings.warn(f'Error policy forward in KGRL: {e}')
            return 0.0
        try:
            action_space_size = pis.shape[-1]
            if torch.any((target_action_tensor < 0) | (target_action_tensor >= action_space_size)):
                raise ValueError(f'KG imitation targets out of bounds for {action_space_size} actions')
            if (pis.gather(1, target_action_tensor.unsqueeze(1)) <= 0).any():
                raise ValueError('KG imitation attempted an invalid masked action')
            dist = torch.distributions.Categorical(probs=pis)
            action_negative_log_probs = -dist.log_prob(target_action_tensor)
            behavior_cloning_loss = action_negative_log_probs.mean()
            if torch.isnan(behavior_cloning_loss) or torch.isinf(behavior_cloning_loss):
                warnings.warn('NaN or Inf imitation loss. Skipping KGRL update.')
                return 0.0
        except ValueError as e:
            warnings.warn(f'Error calculating imitation loss: {e}')
            return 0.0
        except Exception as e:
            warnings.warn(f'Unexpected error calculating imitation loss: {e}')
            return 0.0
        self.optimizer.zero_grad()
        behavior_cloning_loss.backward()
        self.optimizer.step()
        return behavior_cloning_loss.item()

def PPO_initialize():
    ppo = PPO(config=configs)
    return ppo
