# MuJoCo PPO Pusher-v5 on WSL2 + RTX 5070 (8GB)

在 WSL2 + RTX 5070 Laptop (8GB VRAM) 上，用 Stable-Baselines3 的 PPO 算法训练 MuJoCo `Pusher-v5` 任务的最小可复现流程。

## 结果

| 指标 | 数值 |
| :--- | :--- |
| 训练步数 | 50,000 |
| 训练日志 `ep_rew_mean` | 约 -104 |
| 单次确定性评估 `Episode reward` | **-50.85** |
| 随机策略基线 | 约 -100 |

50k 步属于短训练，策略尚未收敛。训练日志中的 `explained_variance` 接近 0，说明价值网络还没学到有效信号；单次评估的 -50.85 与训练日志均值存在差距，且单次评估方差较大，**不代表模型平均水平**。本仓库记录的是“训练→保存→加载→评估”闭环的验证，不是性能复现。

## 环境要求

- WSL2 + Ubuntu 24.04
- NVIDIA 驱动支持 CUDA 12.8+（RTX 5070 为 Blackwell 架构，必须用 cu128 版 PyTorch）
- Conda（Python 3.11）
- 显存 8GB 即可，训练时占用约 2–4GB

## 安装

    conda create -n vla-test python=3.11 -y
    conda activate vla-test

    # Blackwell 显卡必须使用 cu128
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu128

    pip install "stable-baselines3[extra]==2.9.0" "gymnasium[mujoco]" imageio[ffmpeg]

## WSL2 渲染配置（关键）

WSL2 下 MuJoCo 必须使用 EGL 离屏渲染，不要用 `render_mode="human"`：

    sudo apt update
    sudo apt install -y libglfw3 libglew2.2 libgl1 libglx-mesa0 libosmesa6

    conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl -n vla-test
    conda deactivate && conda activate vla-test

验证：

    python -c "import gymnasium as gym; e=gym.make('Pusher-v5', render_mode='rgb_array'); e.reset(); print(e.render().shape)"
    # 预期: (480, 640, 3)

## 训练

    python train_pusher.py

产出 `ppo_pusher_50k.zip`。

## 评估

    python eval_pusher_50k.py

加载 `ppo_pusher_50k.zip`，输出单次 `Episode reward` 并保存 GIF。

## 踩坑记录

1. **Ubuntu 24.04 包名变化**：`libgl1-mesa-glx` 和 `libglew2.0` 已移除，需替换为 `libgl1`、`libglx-mesa0`、`libglew2.2`。
2. **WSL2 下 `render_mode="human"` 不稳定**：统一改用 `rgb_array` + `imageio` 保存 GIF 查看。
3. **预训练模型 Hub 加载依赖冲突**：`huggingface-sb3` / `sb3-team-hf-integration` 与新版 `huggingface_hub` 存在版本冲突，最终跳过 Hub，直接自己训练。
4. **50k 步不足以收敛**：Pusher-v5 官方基线通常训练到 1M 步量级。本仓库记录的是闭环验证，不是性能复现。

## 下一步改进方向

- 加入 `VecNormalize(norm_obs=True, norm_reward=True)`：Pusher 的观测数值范围差异大，不做归一化时 `explained_variance` 长期接近 0，价值网络难以学习。
- 将训练步数提升到 1M 量级，并做 10 次以上评估报告均值±标准差，而不是单次结果。

## License

MIT
