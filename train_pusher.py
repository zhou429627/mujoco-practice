from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

N_ENVS = 4
env = make_vec_env("Pusher-v5", n_envs=N_ENVS)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
)

model.learn(total_timesteps=50_000)
model.save("ppo_pusher_50k")
print("Saved ppo_pusher_50k.zip")
env.close()
