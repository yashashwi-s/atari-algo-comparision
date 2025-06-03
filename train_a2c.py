import os
import torch
import gymnasium as gym
from stable_baselines3 import A2C
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

def train_a2c(config_path):
    print("="*50)
    print("STARTING A2C TRAINING")
    print("="*50)
    
    config = load_config(config_path)
    
    log_dir, save_path = setup_directories(config)
    
    print(f"\nConfiguration:")
    print(f"  Environment: {config['env_id']}")
    print(f"  Total timesteps: {config['total_timesteps']}")
    print(f"  Log directory: {log_dir}")
    print(f"  Save path: {save_path}")
    print(f"  Number of environments: {config['n_envs']}")
    
    use_wandb = setup_wandb(config, is_training=True)
    
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path=save_path,
        name_prefix='a2c_checkpoint',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    print(f"Checkpoint callback added (save every {config['save_freq']} steps)")
    
    print(f"\nCreating {config['n_envs']} parallel environments...")
    env_fns = [make_env(config['env_id'], seed=config['seed'] + i, log_dir=log_dir) 
              for i in range(config['n_envs'])]
    env = DummyVecEnv(env_fns)
    print(f"DummyVecEnv created with {len(env_fns)} environments")
    
    env = VecFrameStack(env, n_stack=config['n_stack'])
    print(f"VecFrameStack applied with n_stack={config['n_stack']}")
    
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
    
    print(f"\nCreating A2C model...")
    print(f"  Policy: {config['policy']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=config['gradient_save_freq'],
            model_save_path=save_path,
            verbose=1
        )
        callbacks.append(wandb_callback)
        print("W&B callback added")
    
    model = A2C(
        policy=config['policy'],
        env=env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        use_rms_prop=config['use_rms_prop'],
        rms_prop_eps=config['rms_prop_eps'],
        normalize_advantage=config['normalize_advantage'],
        stats_window_size=config['stats_window_size'],
        tensorboard_log=log_dir,
        policy_kwargs=config['policy_kwargs'],
        verbose=config['verbose'],
        seed=config['seed'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print(f"A2C model created successfully on device: {model.device}")
    
    if config['pretrained']:
        print(f"\nLoading pretrained model from: {config['saved_model_path']}")
        try:
            model = A2C.load(
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
    train_a2c("configs/a2c_config.yaml") 
