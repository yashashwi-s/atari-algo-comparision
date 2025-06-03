import os
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback
import wandb

from utils import (
    make_env, 
    unzip_file, 
    load_config, 
    setup_wandb,
    setup_directories
)

def train_dqn(config_path):
    config = load_config(config_path)
    
    log_dir, save_path = setup_directories(config)
    
    print("="*50)
    print("STARTING DQN TRAINING")
    print("="*50)
    
    print(f"\nConfiguration:")
    print(f"  Environment: {config['env_id']}")
    print(f"  Total timesteps: {config['total_timesteps']}")
    print(f"  Log directory: {log_dir}")
    print(f"  Save path: {save_path}")
    
    use_wandb = setup_wandb(config, is_training=True)
    
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path=save_path,
        name_prefix='dqn_checkpoint',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    print(f"Checkpoint callback added (save every {config['save_freq']} steps)")
    
    print(f"\nCreating environment...")
    env_fns = [make_env(config['env_id'], seed=config['seed'], log_dir=log_dir)]
    env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=config['n_stack'])
    print(f"Environment created with proper monitoring for reward tracking")
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_path,
        log_path=log_dir,
        eval_freq=config['check_freq'],
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    print(f"Evaluation callback added (evaluate every {config['check_freq']} steps)")
    
    if use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=config['gradient_save_freq'],
            model_save_path=save_path,
            verbose=1
        )
        callbacks.append(wandb_callback)
        print("W&B callback added")
    
    print(f"\nCreating DQN model...")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    model = DQN(
        policy=config['policy'],
        env=env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        target_update_interval=config['target_update_interval'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        max_grad_norm=config['max_grad_norm'],
        stats_window_size=config['stats_window_size'],
        tensorboard_log=log_dir,
        policy_kwargs=config['policy_kwargs'],
        verbose=config['verbose'],
        seed=config['seed'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print(f"Model running on device: {model.device}")
    
    if config['pretrained']:
        print(f"\nLoading pretrained model from: {config['saved_model_path']}")
        try:
            model = DQN.load(
                config['saved_model_path'], 
                env=env, 
                verbose=config['verbose'], 
                tensorboard_log=log_dir
            )
            unzip_file(config['saved_model_path'], config['unzip_file_path'])
            model.policy.load_state_dict(
                torch.load(os.path.join(config['unzip_file_path'], "policy.pth"))
            )
            model.policy.optimizer.load_state_dict(
                torch.load(os.path.join(config['unzip_file_path'], "policy.optimizer.pth"))
            )
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            print("Continuing with fresh model...")
    
    print(f"\n{'='*50}")
    print("STARTING TRAINING")
    print(f"{'='*50}")
    print(f"Total timesteps: {config['total_timesteps']}")
    print(f"Log interval: {config['log_interval']}")
    print(f"Callbacks: {len(callbacks)}")
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        log_interval=config['log_interval'],
        callback=callbacks if callbacks else None,
        progress_bar=True
    )
    
    final_model_path = os.path.join(save_path, 'final_model')
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    env.close()
    if use_wandb:
        wandb.finish()
        print("W&B run finished")
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Check TensorBoard with: tensorboard --logdir={log_dir}")
    if use_wandb:
        print(f"Check W&B dashboard at: https://wandb.ai")


if __name__ == "__main__":
    train_dqn("configs/dqn_config.yaml") 
