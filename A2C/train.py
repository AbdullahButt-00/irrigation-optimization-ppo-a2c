import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
# CHANGED: Import DummyVecEnv instead of SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import warnings
import logging
from aquacropgymnasium.env import Maize

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

output_dir = './train_output'
os.makedirs(output_dir, exist_ok=True)


# ======================= Reward Logging Callback =======================
class RewardLoggingCallback(BaseCallback):
    def __init__(self, agent_name, output_dir, verbose=0, num_intervals=100):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.output_dir = output_dir
        self.num_intervals = num_intervals
        self.episode_rewards = []
        self.current_episode_rewards = []

    def _on_step(self):
        # Get the ORIGINAL (unnormalized) reward from the Monitor wrapper
        infos = self.locals.get('infos', [])
        
        # Check if episode finished and get original reward
        for info in infos:
            if 'episode' in info:
                # This is the REAL reward before normalization
                original_reward = info['episode']['r']
                self.episode_rewards.append(original_reward)
        
        return True

    def _on_training_end(self):
        if self.episode_rewards:
            print(f"Training completed for {self.agent_name}.")
            print(f"  Mean reward: {np.mean(self.episode_rewards):.2f}")
            print(f"  Std reward: {np.std(self.episode_rewards):.2f}")
            print(f"  Min reward: {np.min(self.episode_rewards):.2f}")
            print(f"  Max reward: {np.max(self.episode_rewards):.2f}")
        self.plot_rewards()

    def plot_rewards(self):
        rewards = np.array(self.episode_rewards)
        if len(rewards) == 0:
            return

        # Create MUCH TALLER figure for better y-axis visibility
        plt.figure(figsize=(16, 12))
        
        if len(rewards) >= self.num_intervals:
            interval_size = len(rewards) // self.num_intervals
            avg_rewards = np.array([
                np.mean(rewards[i * interval_size:(i + 1) * interval_size])
                for i in range(self.num_intervals)
            ])
            x = np.arange(self.num_intervals) * interval_size
            plt.plot(x, avg_rewards, marker='o', linewidth=2, markersize=4, color='#2E86AB')
            plt.xlabel('Episodes', fontsize=16, fontweight='bold')
            plt.ylabel('Average Total Reward (Actual)', fontsize=16, fontweight='bold')
            plt.title(f'Training Rewards - {self.agent_name}', fontsize=18, fontweight='bold', pad=20)
        else:
            x = np.arange(len(rewards))
            plt.plot(x, rewards, linewidth=1.5, alpha=0.7, color='#2E86AB')
            plt.xlabel('Episodes', fontsize=16, fontweight='bold')
            plt.ylabel('Total Reward (Actual)', fontsize=16, fontweight='bold')
            plt.title(f'Training Rewards - {self.agent_name}', fontsize=18, fontweight='bold', pad=20)
        
        # Calculate min and max with some padding for better visibility
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        reward_range = max_reward - min_reward
        padding = reward_range * 0.1  # 10% padding
        
        # Set y-limits with padding to ensure full visibility
        plt.ylim(min_reward - padding, max_reward + padding)
        
        # Improve y-axis formatting
        plt.ticklabel_format(style='plain', axis='y')
        plt.tick_params(axis='both', labelsize=13)
        
        # Add horizontal lines at min and max for reference
        plt.axhline(y=min_reward, color='red', linestyle='--', alpha=0.3, linewidth=1, label=f'Min: {min_reward:.2f}')
        plt.axhline(y=max_reward, color='green', linestyle='--', alpha=0.3, linewidth=1, label=f'Max: {max_reward:.2f}')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = f'Mean: {np.mean(rewards):.2f}\nStd: {np.std(rewards):.2f}\nMin: {min_reward:.2f}\nMax: {max_reward:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
        
        # Add legend
        plt.legend(loc='lower right', fontsize=11)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save with high DPI
        plt.savefig(os.path.join(self.output_dir, f'reward_plot_{self.agent_name}.png'), 
                    format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save reward data for later re-plotting
        np.save(os.path.join(self.output_dir, f'rewards_{self.agent_name}.npy'), rewards)
        print(f"Rewards data saved to: rewards_{self.agent_name}.npy")


# ======================= Environment Creator =======================
def make_env():
    def _init():
        env = Maize(mode='train', year1=1982, year2=2007)
        return Monitor(env)
    return _init


# ======================= MAIN EXECUTION =======================
if __name__ == "__main__":

    # Lowered to 8 for stability (Runs sequentially now)
    NUM_ENVS = 8
    timestep_values = [2500000]

    for train_timesteps in timestep_values:

        # CHANGED: Use DummyVecEnv. This runs in the SAME process (saves RAM)
        # instead of creating new processes like SubprocVecEnv did.
        print(f"DEBUG: Current train_timesteps = {train_timesteps}")
        train_env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        model_name = f"a2c_model_{train_timesteps}"

        callback = RewardLoggingCallback(agent_name=model_name, output_dir=output_dir)

        model = A2C(
            "MlpPolicy",
            train_env,
            device="cpu",        # A2C is usually faster on CPU for this environment size
            learning_rate=7e-4, 
            n_steps=5,           
            gamma=0.99,
            ent_coef=0.001,      
            verbose=1,
            tensorboard_log=os.path.join(output_dir, f"tensorboard_logs_{train_timesteps}")
        )

        print(f"Training {model_name} for {train_timesteps} timesteps...")
        model.learn(total_timesteps=train_timesteps, callback=callback)

        model.save(os.path.join(output_dir, f"{model_name}.zip"))
        train_env.save(os.path.join(output_dir, f"{model_name}_vecnormalize.pkl"))
        train_env.close()

        print(f"Training completed for {model_name}!")