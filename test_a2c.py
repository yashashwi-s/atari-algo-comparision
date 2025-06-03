#!/usr/bin/env python3

import os
import torch
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import wandb

from utils import (
    make_env, 
    unzip_file, 
    load_config, 
    setup_wandb
)

def test_a2c(config_path):
    config = load_config(config_path)
    
    use_wandb = setup_wandb(config, is_training=False)
    
    if config['pretrained']:
        unzip_file(config['saved_model_path'], config['unzip_file_path'])
    
    env = DummyVecEnv([make_env(config['env_id'], seed=config['seed'], render_mode='rgb_array')])
    env = VecFrameStack(env, n_stack=config['n_stack'])
    
    model = A2C(
        policy=config['policy'],
        env=env,
        verbose=config['verbose']
    )
    
    if config['pretrained']:
        model.policy.load_state_dict(
            torch.load(os.path.join(config['unzip_file_path'], "policy.pth"))
        )
        model.policy.optimizer.load_state_dict(
            torch.load(os.path.join(config['unzip_file_path'], "policy.optimizer.pth"))
        )
    else:
        model = A2C.load(
            config['saved_model_path'][:-4], 
            env=env
        )
    
    all_rewards = []
    
    for episode in range(config['test_episodes']):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            env.render()
            
            done = done[0]
            
        all_rewards.append(episode_reward)
        
        if use_wandb:
            wandb.log({
                'test_episode_reward': episode_reward, 
                'test_episode': episode
            })
            
        print(f"Episode {episode+1}/{config['test_episodes']}, Reward: {episode_reward}")
    
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"\n===== Test Results =====")
    print(f"Average Reward over {config['test_episodes']} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {np.min(all_rewards):.2f}, Max Reward: {np.max(all_rewards):.2f}")
    
    if use_wandb:
        wandb.log({
            'average_test_reward': avg_reward,
            'std_test_reward': std_reward,
            'min_test_reward': np.min(all_rewards),
            'max_test_reward': np.max(all_rewards)
        })
    
    env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    test_a2c("configs/a2c_config.yaml") 
