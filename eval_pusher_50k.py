import gymnasium as gym
import imageio
from stable_baselines3 import PPO

env = gym.make("Pusher-v5", render_mode="rgb_array")
model = PPO.load("ppo_pusher_50k")   # 加载自己训练的模型

obs, _ = env.reset()
frames = []
total_reward = 0.0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    frames.append(env.render())
    done = terminated or truncated

env.close()
imageio.mimsave("pusher_50k.gif", frames, fps=30)
print(f"Episode reward: {total_reward:.2f}")
print(f"Saved {len(frames)} frames to pusher_50k.gif")
