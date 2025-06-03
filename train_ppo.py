import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from utils import (
    make_env,
    unzip_file,
    load_config,
    setup_wandb,
    setup_directories
)


def train_ppo(config_path):
    print("="*50)
    print("STARTING PPO TRAINING")
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
    
    print(f"\nCreating {config['n_envs']} parallel environments...")
    env_fns = [make_env(config['env_id'], seed=config['seed'] + i, log_dir=log_dir) 
               for i in range(config['n_envs'])]
    
    env = DummyVecEnv(env_fns)
    print(f"DummyVecEnv created with {len(env_fns)} environments")
    
    env = VecFrameStack(env, n_stack=config['n_stack'])
    print(f"VecFrameStack applied with n_stack={config['n_stack']}")
    
    print(f"\nCreating PPO model...")
    print(f"  Policy: {config['policy']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  TensorBoard log dir: {log_dir}")
    
    model = PPO(
        policy=config['policy'],
        env=env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        clip_range_vf=config['clip_range_vf'],
        normalize_advantage=config['normalize_advantage'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        use_sde=config['use_sde'],
        sde_sample_freq=config['sde_sample_freq'],
        target_kl=config['target_kl'],
        stats_window_size=config['stats_window_size'],
        tensorboard_log=log_dir,
        policy_kwargs=config['policy_kwargs'],
        verbose=config['verbose'],
        seed=config['seed'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print(f"PPO model created successfully on device: {model.device}")
    
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path=save_path,
        name_prefix='ppo_checkpoint',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    print(f"Checkpoint callback added (save every {config['save_freq']} steps)")
    
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
    
    if config['pretrained']:
        print(f"\nLoading pretrained model from: {config['saved_model_path']}")
        try:
            model = PPO.load(
                config['saved_model_path'],
                env=env,
                tensorboard_log=log_dir
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
    
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            log_interval=config['log_interval'],
            callback=callbacks if callbacks else None,
            progress_bar=True
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    
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
    train_ppo("configs/ppo_config.yaml")
