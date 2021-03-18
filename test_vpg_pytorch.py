#!/usr/bin/env python
import sys
from pathlib import Path
import torch
from vpg_pytorch import MLP
import gym
import numpy as np


env_name='CartPole-v0'
def main():    
    if len(sys.argv) < 2:
        print('Need the path of the directory containing learnt weight')
        return
    print(sys.argv)
    path = Path(sys.argv[1])
    if not path.exists():
        print('The given path not exist')
        return
    
    if not path.is_dir():
        print('The given path not directory')
        return
    
    env = gym.make(env_name)
    model = MLP([env.observation_space.shape[0], 32, 20, env.action_space.n])
    action_space = np.arange(env.action_space.n)
    for sub_path in path.glob('vpg_weight_ep*.pth'):
        if not sub_path.is_file():
            print("%r not file" % sub_path)
            continue
        model.load_state_dict(torch.load(sub_path))
        model.eval()
        reward_tot = 0

        print("File = %r" % sub_path)

        for ep in range(100):
            state = env.reset()
            done = False
            reward_for_ep = 0
            while not done:
                action_probs = model(torch.FloatTensor(state)).detach().numpy()
                action = np.random.choice(action_space, p=action_probs)
                state, r, done, _ = env.step(action)
                reward_for_ep += r # actually, reward = 1 for each step
            reward_tot += reward_for_ep
            # print("\tEp. = %d, reward = %d" % (ep, reward_for_ep))
        print("File = %r, avg reward = %f, is_success = %r" % (sub_path, reward_tot / 100.0, reward_tot >= 19500))

if __name__ == '__main__':
    main()