import torch
import torch.nn as nn

class TransformerActor(nn.Module):
    def __init__(self, d_node, d_task, max_tasks, m1, m2, d_model, nhead, num_layers):
        super(TransformerActor, self).__init__()
        self.d_node = d_node
        self.d_task = d_task
        self.max_tasks = max_tasks
        self.m1 = m1  # Maximum task nodes
        self.m2 = m2  # Maximum compute nodes
        self.d_model = d_model

        # Embedding layers
        self.task_node_embedding = nn.Linear(d_node, d_model)
        self.task_data_embedding = nn.Linear(d_task, d_model)
        self.compute_node_embedding = nn.Linear(d_node, d_model)

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

        # Compute node selection head sigmoid for possible values [0, 1]
        self.compute_node_selector = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, task_node_self, task_data_self, neighbor_task_node, neighbor_task_data, neighbor_compute_node, task_self_mask, neighbor_task_mask, neighbor_compute_node_mask):
        """
        Args:
            task_node_self: Tensor of shape [batch_size, d_node]
            task_data_self: Tensor of shape [batch_size, max_tasks, d_task]
            neighbor_task_node: Tensor of shape [batch_size, m1, d_node]
            neighbor_task_data: Tensor of shape [batch_size, m1, max_tasks, d_task]
            neighbor_compute_node: Tensor of shape [batch_size, m2, d_node]
            task_self_mask: Tensor of shape [batch_size, max_tasks] (1 for valid task nodes, 0 for padding)
            neighbor_task_mask: Tensor of shape [batch_size, m1, max_tasks] (1 for valid tasks, 0 for padding)
            neighbor_compute_node_mask: Tensor of shape [batch_size, m2] (1 for valid compute nodes, 0 for padding)
        """
        batch_size = task_node_self.size(0)

        # Process self task node and data
        task_node_self_emb = self.task_node_embedding(task_node_self).unsqueeze(1)  # [batch_size, 1, d_model]
        task_data_self_emb = self.task_data_embedding(task_data_self)  # [batch_size, max_tasks, d_model]
        task_data_self_combined = task_data_self_emb + task_node_self_emb  # Add task node embedding to each task
        task_data_self_sequence = task_data_self_combined.view(batch_size, -1, self.d_model)  # Flatten to [batch_size, max_tasks, d_model]

        # Process neighbor task nodes and data
        neighbor_task_node_emb = self.task_node_embedding(neighbor_task_node)  # [batch_size, m1, d_model]
        neighbor_task_data_emb = self.task_data_embedding(neighbor_task_data)  # [batch_size, m1, max_tasks, d_model]
        neighbor_task_data_combined = neighbor_task_data_emb + neighbor_task_node_emb.unsqueeze(2)  # Add node embedding to each task
        neighbor_task_data_sequence = neighbor_task_data_combined.view(batch_size, -1, self.d_model)  # Flatten to [batch_size, m1 * max_tasks, d_model]

        # Process neighbor compute nodes
        neighbor_compute_node_emb = self.compute_node_embedding(neighbor_compute_node)  # [batch_size, m2, d_model]

        # Combine all sequences
        combined_sequence = torch.cat([
            task_data_self_sequence,  # Self tasks [batch_size, max_tasks, d_model]
            neighbor_task_data_sequence,  # Neighbor tasks [batch_size, m1 * max_tasks, d_model]
            neighbor_compute_node_emb  # Compute nodes [batch_size, m2, d_model]
        ], dim=1)  # [batch_size, max_tasks + m1 * max_tasks + m2, d_model]

        # Create combined mask
        neighbor_task_mask_flat = neighbor_task_mask.view(batch_size, -1)  # [batch_size, m1 * max_tasks]
        combined_mask = torch.cat([
            task_self_mask,  # Self task mask
            neighbor_task_mask_flat,  # Neighbor task mask
            neighbor_compute_node_mask  # Compute node mask
        ], dim=1)  # [batch_size, max_tasks + m1 * max_tasks + m2]

        # Apply Transformer Encoder
        transformer_output = self.transformer(
            src=combined_sequence.permute(1, 0, 2),  # [seq_len, batch_size, d_model]
            src_key_padding_mask=(1 - combined_mask).bool()  # Mask padded positions
        ).permute(1, 0, 2)  # [batch_size, seq_len, d_model]

        # Extract compute node outputs
        compute_outputs = transformer_output[:, -self.m2:, :]  # [batch_size, m2, d_model]
        local_output = transformer_output[:, :self.max_tasks, :].mean(dim=1, keepdim=True)  # Aggregate self task embeddings to represent local computation

        # Combine local computation with compute node outputs
        final_outputs = torch.cat([local_output, compute_outputs], dim=1)  # [batch_size, m2 + 1, d_model]

        # Select compute node or local computation
        compute_posssibility = self.compute_node_selector(final_outputs).squeeze(-1) # [batch_size, m2 + 1]
        
        return compute_posssibility

class MultiAgentTransformerCritic(nn.Module):
    def __init__(self, d_node, d_task, max_tasks, max_agents, m1, m2, d_model, nhead, num_layers):
        super(MultiAgentTransformerCritic, self).__init__()
        self.d_node = d_node
        self.d_task = d_task
        self.max_tasks = max_tasks
        self.max_agents = max_agents
        self.d_model = d_model
        self.m1 = m1  # Maximum task nodes for each agent
        self.m2 = m2  # Maximum compute nodes for each agent
        self.d_action = m2+1 # Compute node selection head output + 1 for local computation

        # Embedding layers
        self.task_node_embedding = nn.Linear(d_node, d_model)
        self.task_data_embedding = nn.Linear(d_task, d_model)

        # Transformer Encoder for joint agent representation
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

        # Q-value prediction
        self.q_value_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # Output Q-value
        )

    def forward(self, task_node_agent_self, task_data_agent_self, neighbor_task_node, neighbor_task_data, neighbor_compute_node, agent_action, task_self_mask, neighbor_task_mask, neighbor_compute_node_mask):
        """
        Args:
            task_node_agent_self: Tensor of shape [batch_size, max_agents, d_node]
            task_data_agent_self: Tensor of shape [batch_size, max_agents, max_tasks, d_task]
            neighbor_task_node: Tensor of shape [batch_size, max_agents, m1, d_node]
            neighbor_task_data: Tensor of shape [batch_size, max_agents, m1, max_tasks, d_task]
            neighbor_compute_node: Tensor of shape [batch_size, max_agents, m2, d_node]
            agent_action: Tensor of shape [batch_size, max_agents, d_action]
            task_self_mask: Tensor of shape [batch_size, max_agents, max_tasks] (1 for valid tasks, 0 for padding)
            neighbor_task_mask: Tensor of shape [batch_size, max_agents, m1, max_tasks] (1 for valid tasks, 0 for padding)
            neighbor_compute_node_mask: Tensor of shape [batch_size, max_agents, m2] (1 for valid compute nodes, 0 for padding)

        Returns:
            q_value: Tensor of shape [batch_size, 1] (Q-value for all agents)
        """
        # Process agent task nodes and data
        task_node_emb = self.task_node_embedding(task_node_agent_self).unsqueeze(2)  # [batch_size, max_agents, 1, d_model]
        task_data_emb = self.task_data_embedding(task_data_agent_self)  # [batch_size, max_agents, max_tasks, d_model]
        task_data_agentself_combined = task_data_emb + task_node_emb  # [batch_size, max_agents, max_tasks, d_model]
        task_data_agentself_sequence = task_data_agentself_combined.view(-1, self.max_agents, self.max_tasks, self.d_model)  # Flatten to [batch_size, max_agents, max_tasks, d_model]

        # Process neighbor task nodes and data
        neighbor_task_node_emb = self.task_node_embedding(neighbor_task_node).unsqueeze(3)  # [batch_size, max_agents, m1, 1, d_model]
        neighbor_task_data_emb = self.task_data_embedding(neighbor_task_data)  # [batch_size, max_agents, m1, max_tasks, d_model]
        neighbor_task_data_combined = neighbor_task_data_emb + neighbor_task_node_emb  # [batch_size, max_agents, m1, max_tasks, d_model]
        neighbor_task_data_sequence = neighbor_task_data_combined.view(-1, self.max_agents, self.max_tasks * self.m1, self.d_model)  # Flatten to [batch_size, max_agents, max_tasks * m1, d_model]

        # Process neighbor compute nodes
        neighbor_compute_node_emb = self.task_node_embedding(neighbor_compute_node) # [batch_size, max_agents, m2, d_model]

        # Combine agent task data, neighbor task data, and neighbor compute nodes
        combined_emb = torch.cat([
            task_data_agentself_sequence,  # Self tasks [batch_size, max_agents, max_tasks, d_model]
            neighbor_task_data_sequence,  # Neighbor tasks [batch_size, max_agents, max_tasks * m1, d_model]
            neighbor_compute_node_emb  # Compute nodes [batch_size, max_agents, m2, d_model]
        ], dim=2) # [batch_size, max_agents, max_tasks + max_tasks * m1 + m2, d_model]

        # Create combined mask
        neighbor_task_mask_flat = neighbor_task_mask.view(-1, self.max_agents, self.max_tasks * self.m1) # [batch_size, max_agents, max_tasks * m1]
        combined_mask = torch.cat([
            task_self_mask,  # Self task mask
            neighbor_task_mask_flat,  # Neighbor task mask
            neighbor_compute_node_mask  # Compute node mask
        ], dim=2) # [batch_size, max_agents, max_tasks + max_tasks * m1 + m2]

        combined_emb_3d = combined_emb.view(-1, self.max_agents*(self.max_tasks + self.max_tasks * self.m1 + self.m2), self.d_model)  # [batch_size, max_agents * (max_tasks + max_tasks * m1 + m2), d_model]
        combined_mask_2d = combined_mask.view(-1, self.max_agents * self.max_tasks + self.max_tasks * self.m1 + self.m2)  # [batch_size, max_agents * (max_tasks + max_tasks * m1 + m2)]
        # Apply Transformer Encoder for joint agent representation
        transformer_output = self.transformer(
            src=combined_emb_3d.permute(1, 0, 2),  # [max_agents * (max_tasks + max_tasks * m1 + m2), batch_size, d_model]
            src_key_padding_mask=(1 - combined_mask_2d).bool()  # Mask invalid agents
        ).permute(1, 0, 2) # [batch_size, max_agents * (max_tasks + max_tasks * m1 + m2), d_model]

        # Select agent actions for each agent from transformer output
        transformer_output = transformer_output.view(-1, self.max_agents, self.max_tasks + self.max_tasks * self.m1 + self.m2, self.d_model)  # [batch_size, max_agents, max_tasks + max_tasks * m1 + m2, d_model]
        agent_compute_outputs = transformer_output[:, :, -self.m2:, :]  # [batch_size, max_agents, m2, d_model]
        agent_local_outputs = transformer_output[:, :, :self.max_tasks, :].mean(dim=2, keepdim=True)  # Aggregate self task embeddings to represent local computation, [batch_size, max_agents, 1, d_model]
        agent_evaluation_outputs = torch.cat([agent_local_outputs, agent_compute_outputs], dim=2)  # [batch_size, max_agents, m2 + 1, d_model]
        agent_q_values = self.q_value_predictor(agent_evaluation_outputs).squeeze(-1)  # [batch_size, max_agents, m2 + 1]

        # Use action tensor to select Q-values, sum for total Q-value
        agent_action_prob = agent_action_prob.view(-1, self.max_agents, self.d_action) # [batch_size, max_agents, m2 + 1]
        q_value = (agent_q_values * agent_action_prob).sum(dim=2) # [batch_size, max_agents]
        total_q_value = q_value.sum(dim=1, keepdim=True)  # [batch_size, 1]
        return total_q_value
