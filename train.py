import os
import torch
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.vision_surface_env import VisionSurfaceTracingEnv
from models.cnn_policy import CustomCNNWithFusion 
from stable_baselines3.common.logger import configure

# Configure custom CNN feature extractor for visual input
policy_kwargs = dict(
    features_extractor_class=CustomCNNWithFusion,
    features_extractor_kwargs=dict(features_dim=256)
)

# Choose GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Environment factory function
def make_env():
    def _init():
        env = VisionSurfaceTracingEnv(
            pcd_path=None,                    
            auto_switch_pcd=True            
        )
        return env
    return _init

# Wrap the environment in a vectorized format (needed by SB3 PPO)
env = DummyVecEnv([make_env()])
print("Environment loaded with surface coverage-based reward.")

# Set up logging directory for TensorBoard
log_dir = "./ppo_surface_tensorboard"
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Initialize PPO model with customized policy and training hyperparameters
model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,      
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    device=device,
    tensorboard_log=log_dir,
)

model.learn(total_timesteps=1000000)

# Save the trained model
model.save("models/ppo_surface.zip")
print("Model saved to models/ppo_surface.zip")
