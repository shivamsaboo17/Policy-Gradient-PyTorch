#!/usr/bin/env python
import numpy as np
import gym
import torch.nn as nn
import torch
import torch.optim as optimizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
import gin.torch


class MLP(nn.Module):
    """
    Returns a fully connected neural network of 
    given dimensions. The input is of dimensions
    of the shape of the observations and output
    is of the shape of number of actions.
    """

    def __init__(self, sizes, activation=nn.ReLU(inplace=True), output_activation=None):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation)
        self.fwd = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.fwd(x), dim=-1)


def discount_rewards(reward, gamma=0.99):
    # Cumulative discounted sum
    r = np.array([gamma ** i * reward[i]
                  for i in range(len(reward))])
    r = r[::-1].cumsum()[::-1]
    # Subtracting the baseline reward
    # Intuitively this means if the network predicts what it
    # expects it should not do too much about it
    # Stabalizes and speeds up the training
    return r - r.mean()


@gin.configurable
def train(env_id='CartPole-v0', save_dir=None, num_episodes=2000, batch_size=10, lr=1e-2, gamma=0.99, cp_path=None):
    if save_dir is None:
        save_dir = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        Path(f'Save/{save_dir}').mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id)
    op = getattr(env, "getLogger", None)
    if op is not None:
        logger = env.getLogger()
    else:
        logger = logging.getLogger('vpg')

    model = MLP([env.observation_space.shape[0], 32, 20, env.action_space.n])
    opt = optimizer.Adam(model.parameters(), lr)
    prev_episode = 0
    if cp_path is not None:
        checkpoint = torch.load(cp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_episode = checkpoint['episode']
        logger.info(f"Loaded successfully checkpoint from file {cp_path}")
        logger.info(f"Restarting from {prev_episode} th epsiode...")

    action_space = np.arange(env.action_space.n)

    # Lists for maintaing logs
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []

    batch_counter = 0

    for ep in range(prev_episode, num_episodes):
        # Reset
        s_0 = env.reset()
        states = []
        reward = []
        actions = []
        complete = False
        while not complete:
            action_probs = model(torch.FloatTensor(s_0)).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s1, r, complete, _ = env.step(action)
            states.append(s_0)
            reward.append(r)
            actions.append(action)
            s_0 = s1
            if complete:
                logger.info(f"Ep {ep + 1}, step_cnt = {len(states)}")
                batch_rewards.extend(discount_rewards(reward, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(reward))

                if batch_counter == batch_size:
                    logger.info(f"To update. Last 100 ep. avg. reward = {np.mean(total_rewards[-100:])}.")
                    # Save the last trained model's weights if they are good
                    if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= 20000:
                        filename = 'Save/%s/vpg_weight_ep%d.pth' % (save_dir, ep + 1)
                        logger.info(f"Save weight for eval: path = {filename}")
                        torch.save(
                            model.state_dict(),
                            filename
                        )
                    # Prepare the batches for training
                    # Add states, reward and actions to tensor
                    opt.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    action_tensor = torch.LongTensor(batch_actions)

                    # Convert the probs by the model to log probabilities
                    log_probs = torch.log(model(state_tensor))
                    # Mask the probs of the selected actions
                    selected_log_probs = reward_tensor * log_probs[np.arange(len(action_tensor)), action_tensor]
                    # Loss is negative of expected policy function J = R * log_prob
                    loss = -selected_log_probs.mean()

                    # Do the update gradient descent(with negative reward hence is gradient ascent)
                    loss.backward()
                    opt.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 0

                    # Save the current learning info as a checkpoint for every 1000 Ep.
                    if ep % 50 == 49:
                        path = f'Save/{save_dir}/vpg_cp_ep_{ep + 1}.pth'
                        torch.save({
                            'episode': ep + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'loss': loss
                        }, path)
                        logger.info(f"Checkpoint saved to file {path}")

    return total_rewards


if __name__ == '__main__':
    start_time_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    Path('Save/%s' % start_time_str).mkdir(parents=True, exist_ok=True)
    gin.parse_config_file('config.gin')
    rew = train()
    # env_name = 'gym_rds_cartpole:RdsCartPole-async-v0'
    # rew = train(env_id=env_name, save_dir=start_time_str, num_episodes=10000)
    window = 10
    smoothed_rewards = [np.mean(rew[i - window:i + 1]) if i > window
                        else np.mean(rew[:i + 1]) for i in range(len(rew))]

    plt.figure(figsize=(12, 8))
    plt.plot(rew)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.savefig('Save/%s/figure.png' % start_time_str)
    plt.show()
