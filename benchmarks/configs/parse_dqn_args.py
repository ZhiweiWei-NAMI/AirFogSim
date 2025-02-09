
import argparse
import os

def parseDQNArgs():
    parser = argparse.ArgumentParser(description='DQN arguments')
    parser.add_argument('--d_node', type=int, default=6)
    parser.add_argument('--d_task', type=int, default=7)
    parser.add_argument('--max_tasks', type=int, default=3)
    parser.add_argument('--m1', type=int, default=50)
    parser.add_argument('--m2', type=int, default=50)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:3')
    # epsilon
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--replay_buffer_capacity', type=int, default=10000)
    parser.add_argument('--replay_buffer_update_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    # args.model_dir
    parser.add_argument('--model_dir', type=str, default='models/simple_dqn/')
    parser.add_argument('--model_path', type=str, default='models/simple_dqn/model_499968.pth')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_model_freq', type=int, default=100000)

    parser.add_argument('--reschedule', type=bool, default=False)
    # mobility_mask
    parser.add_argument('--mobility_mask', type=bool, default=False)

    args = parser.parse_args()
    # if model_dir not exists, create it
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    return args