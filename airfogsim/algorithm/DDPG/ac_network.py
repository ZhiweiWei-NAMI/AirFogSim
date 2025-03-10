
import torch
from torch import nn
from torch.nn import functional as F

class ActorNetwork(nn.Module):
    def __init__(self, d_node, d_task, max_tasks, m1, m2, d_model):
        super(ActorNetwork, self).__init__()
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

        # full connected layer
        self.fc = nn.Linear(d_model * (m1 * max_tasks + m2), 4 * d_model)

        # Compute node selection head sigmoid for possible values [0, 1]
        self.compute_node_selector = nn.Sequential(
            nn.Linear(4 * d_model, 2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model, (self.m2+1)*(self.m1*max_tasks))
        )

    def forward(self, task_node, task_data, compute_node, task_mask, compute_node_mask):
        """
        Args:
            task_node: [batch_size, m1, d_node]
            task_data: [batch_size, m1, max_tasks, d_task]
            compute_node: [batch_size, m2, d_node]
            task_mask: [batch_size, m1, max_tasks], 1 for valid tasks, 0 for padding
            compute_node_mask: [batch_size, m2]

        Returns:
            valid_compute_score: [batch_size, m1 * max_tasks, m2]
        """
        batch_size = task_node.size(0)

        # Process neighbor task nodes and data
        task_node_emb = self.task_node_embedding(task_node)  # [batch_size, m1, d_model]
        task_data_emb = self.task_data_embedding(task_data)  # [batch_size, m1, max_tasks, d_model]
        task_data_combined = task_node_emb.unsqueeze(2) + task_data_emb  # Add node embedding to each task
        task_data_sequence = task_data_combined.view(batch_size, -1, self.d_model)  # Flatten to [batch_size, m1 * max_tasks, d_model]

        # Process neighbor compute nodes
        compute_node_emb = self.compute_node_embedding(compute_node)  # [batch_size, m2, d_model]

        # Combine all sequences
        combined_sequence = torch.cat([
            task_data_sequence,  # tasks [batch_size, m1 * max_tasks, d_model]
            compute_node_emb  # Compute nodes [batch_size, m2, d_model]
        ], dim=1)  # [batch_size, m1 * max_tasks + m2, d_model]

        combined_sequence = combined_sequence.view(batch_size, -1)  # [batch_size, m1 * max_tasks + m2, d_model]

        # Apply fully connected layer
        output = self.fc(combined_sequence) # [batch_size, 4 * d_model]

        # Select compute node or local computation
        compute_score = self.compute_node_selector(output) # [batch_size, (m2+1) * m1 * max_tasks]
        compute_score = compute_score.view(batch_size, self.m1 * self.max_tasks, self.m2+1)
        return compute_score
    
class CriticNetwork(nn.Module):
    def __init__(self, d_node, d_task, max_tasks, m1, m2, d_model):
        super(CriticNetwork, self).__init__()
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

        # full connected layer
        self.fc = nn.Linear(d_model * (m1 * max_tasks + m2) + m1 * max_tasks * (m2+1), 4 * d_model)

        # Compute node selection head sigmoid for possible values [0, 1]
        self.compute_score = nn.Sequential(
            nn.Linear(4 * d_model, 2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model, 1)
        )

    def forward(self, task_node, task_data, compute_node, task_mask, compute_node_mask, action):
        """
        Args:
            task_node: [batch_size, m1, d_node]
            task_data: [batch_size, m1, max_tasks, d_task]
            compute_node: [batch_size, m2, d_node]
            task_mask: [batch_size, m1, max_tasks], 1 for valid tasks, 0 for padding
            compute_node_mask: [batch_size, m2]
            action: [batch_size, m1 * max_tasks * (m2+1)], action for each task

        Returns:
            valid_compute_score: [batch_size, 1]
        """
        batch_size = task_node.size(0)
        action = action.view(batch_size, self.m1 * self.max_tasks * (self.m2+1))
        # Process neighbor task nodes and data
        task_node_emb = self.task_node_embedding(task_node)  # [batch_size, m1, d_model]
        task_data_emb = self.task_data_embedding(task_data)  # [batch_size, m1, max_tasks, d_model]
        task_data_combined = task_node_emb.unsqueeze(2) + task_data_emb  # Add node embedding to each task
        task_data_sequence = task_data_combined.view(batch_size, -1, self.d_model)  # Flatten to [batch_size, m1 * max_tasks, d
        # Process neighbor compute nodes
        compute_node_emb = self.compute_node_embedding(compute_node)
        # Combine all sequences
        combined_sequence = torch.cat([
            task_data_sequence,  # tasks [batch_size, m1 * max_tasks, d_model]
            compute_node_emb  # Compute nodes [batch_size, m2, d_model]
        ], dim=1)
        combined_sequence = combined_sequence.view(batch_size, -1)
        combined_sequence_with_action = torch.cat([combined_sequence, action], dim=1)
        # Apply fully connected layer
        output = self.fc(combined_sequence_with_action)
        q_score = self.compute_score(output) # [batch_size, 1]
        
        return q_score