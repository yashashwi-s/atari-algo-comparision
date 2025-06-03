import gymnasium as gym
import os
import zipfile
import numpy as np
import torch
import yaml
import wandb
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import ale_py

gym.register_envs(ale_py)


def make_env(env_id, seed=0, render_mode=None, log_dir=None):
    def _init():
        print(f"Creating environment: {env_id} with seed: {seed}")
        env = gym.make(env_id, render_mode=render_mode, max_episode_steps=5000)
        print(f"Base environment created: {type(env)}")
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env_idx = len([f for f in os.listdir(log_dir) if f.startswith('monitor_')]) 
            monitor_path = os.path.join(log_dir, f'monitor_{env_idx}')
            env = Monitor(env, monitor_path)
            print(f"Monitor wrapper added with log path: {monitor_path}")
        else:
            env = Monitor(env)
            print(f"Monitor wrapper added without logging")
        
        env = AtariWrapper(env)
        print(f"AtariWrapper added: {type(env)}")
        
        env.reset(seed=seed)
        print(f"Environment reset with seed: {seed}")
        return env
    return _init


def unzip_file(zip_path, extract_to_folder):
    print(f"Unzipping {zip_path} to {extract_to_folder}")
    os.makedirs(extract_to_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
    print("Unzipping completed")


def load_config(config_path):
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Config loaded successfully: {list(config.keys())}")
    return config


def setup_wandb(config, is_training=True):
    print(f"Setting up W&B, log_to_wandb: {config['log_to_wandb']}")
    
    if config['log_to_wandb']:
        project = config['project_train'] if is_training else config['project_test']
        name = config['name_train'] if is_training else config['name_test']
        
        print(f"W&B project: {project}, name: {name}")
        
        try:
            wandb.init(
                project=project,
                entity=config['entity'],
                name=name,
                notes=config['notes'],
                sync_tensorboard=config['sync_tensorboard'],
                config=config
            )
            print("W&B initialized successfully")
            return True
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            print("Continuing without W&B logging")
            return False
    else:
        print("W&B logging disabled in config")
        return False


def setup_directories(config):
    log_dir = config['log_dir']
    save_path = config['save_path']
    
    print(f"Creating directories - log_dir: {log_dir}, save_path: {save_path}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    print("Directories created successfully")
    
    return log_dir, save_path