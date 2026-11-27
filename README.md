# SuperMimic-Q1

基于 PyTorch、MuJoCo 和 MuJoCo Warp 的 Q1 V3 全身动作跟踪训练。
支持单卡、多卡、从零训练、检查点续训、指定动作播放和整段录像。

方法参考 [SuperMimic 论文](https://arxiv.org/abs/2610.28480)，代码独立实现。
当前为研究版本，动作跟踪质量仍在验证。

## 安装

Python 3.13。Linux 使用 NVIDIA CUDA，Apple Silicon 可使用 CPU MuJoCo 和 MPS。

```bash
git clone https://code.iqiyuan-robot.com/sw/SuperMimic-Q1.git
cd SuperMimic-Q1
uv sync --python 3.13 --group dev
uv run --no-sync supermimic --help
```

以下命令在仓库根目录执行。首次 `uv sync` 会下载依赖，完成后再启动训练。
后续命令使用 `--no-sync` 复用已安装的环境。
启动器默认将编译缓存放在本机 `/tmp`；可用 `--compiler-cache-dir` 指定目录。

录像另需安装 FFmpeg；CPU 无屏渲染需要 OSMesa：

```bash
sudo apt-get install ffmpeg libosmesa6
```

## 资产与数据

机器人资产与动作数据集均为本地自备，不随源码分发（`assets/` 与 `*.npz`
已在 `.gitignore` 中忽略）。仓库约定的本地布局如下。

### 机器人资产（Q1 V3）

```text
assets/q1_v3/xml/q1_v3_mc.xml          # 训练/仿真用 MuJoCo 模型
assets/q1_v3/xml/q1_v3.xml             # 基础模型
assets/q1_v3/xml/q1_v3_full_range.xml  # 全关节活动范围
assets/q1_v3/meshes/*.STL              # 24 个连杆网格
assets/q1_v3/urdf/q1_v3*.urdf
assets/q1_v3/config/joint_names_*.yaml
```

### 动作数据集

三套动作库，均为 `.npz`，按动作类别分子目录：

| 数据集 | 子目录（片段数） | 合计 |
|---|---|---:|
| `datasets/motion_amass` | complex 588 · simple 914 · singularity 78 · unreasonable 414 | 1994 |
| `datasets/motion_lafan` | complex 39 · simple 4 · singularity 12 · unreasonable 21 | 76 |
| `datasets/motion_yuanke` | spc_base 27 · zk_base 12 · zk_base2 18 · zw_base 19 | 76 |

每个 `.npz` 的字段：

| 字段 | 内容 |
|---|---|
| `fps` | 采样帧率（amass ≈29.7，lafan/yuanke 为 30） |
| `root_pos` | `[T,3]` 根位置 |
| `root_rot` | `[T,4]` 根旋转四元数 |
| `dof_pos` | `[T,D]` 关节角，`D=24`（amass、lafan）或 `22`（yuanke） |
| `local_body_pos` | `[T,B,3]` 各连杆局部位置，可为空 |
| `link_body_list` | `B` 个连杆名称，可为空 |

含空字段（`local_body_pos`/`link_body_list`）的文件需用
`np.load(path, allow_pickle=True)` 读取。

筛选动作：

```bash
uv run --no-sync supermimic filter   --xml assets/q1_v3/xml/q1_v3_mc.xml   --motion-dir datasets/motion_lafan --output-dir logs/data_screening   --workers 8 --sample-hz 10
```

## 八卡从零训练

```bash
cp configs/q1-fresh.example.toml configs/local-train.toml
# 编辑 robot_xml、motion_dir 和新的 output_dir
uv run --no-sync supermimic launch --config configs/local-train.toml --gpus 8 --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync supermimic launch --config configs/local-train.toml --gpus 8
```

该配置从第 0 轮初始化，训练 250000 轮，开启编译、参考缓存和整段 CUDA graph。
环境数和 batch 都是全局总量，按 GPU 数量均分。

| 配置 | `num_envs` | `model_batch_size` | `policy_batch_size` |
|---|---:|---:|---:|
| 八卡示例 | 8192 | 4096 | 4096 |
| 单卡起步 | 1024 | 512 | 512 |

## 单卡从零训练

```bash
cp configs/q1-fresh-single.example.toml configs/local-train-single.toml
# 编辑 robot_xml、motion_dir 和新的 output_dir
uv run --no-sync supermimic launch --config configs/local-train-single.toml --gpus 1 --dry-run
CUDA_VISIBLE_DEVICES=0 uv run --no-sync supermimic launch --config configs/local-train-single.toml --gpus 1
```

单卡配置使用 1024 环境、model/policy batch 各 512，默认输出到
`logs/q1_fresh_single`。单卡和八卡使用同一训练实现及随机域。

建议在独立 tmux 终端运行。训练配置、逐轮日志、TensorBoard 和检查点保存在
`output_dir`。查看训练记录：

```bash
uv run --no-sync supermimic monitor --run-root logs/q1_fresh
```

## 续训

复制训练配置，保留资产、数据和训练参数，改用新的输出目录并加入：

```toml
resume = "../logs/q1_fresh/checkpoint_5000.pt"
output_dir = "../logs/q1_resume"
iterations = 250000
```

使用相同 GPU 数量启动。搬运检查点时，保留主 `.pt` 及其对应的完整 `.ranks`
目录。`iterations` 是累计训练目标。

## 选择动作播放

```bash
uv run --no-sync supermimic play   --checkpoint logs/q1_fresh/checkpoint_5000.pt   --motion datasets/motion_lafan/complex_motion/run1_subject5.npz   --viewer on
```

省略 `--motion` 时，可用 `--motion-index 0` 选择训练动作。
`--speed 0.5` 半速，`--loops 3` 循环；无显示器时保存轨迹：

```bash
uv run --no-sync supermimic play   --checkpoint logs/q1_fresh/checkpoint_5000.pt   --motion datasets/motion_lafan/complex_motion/run1_subject5.npz   --viewer off --output logs/play_selected.npz
```

## 整段录像

录像使用保留全部分段的原始 LAFAN 目录，输出策略与参考并排的 MP4。

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa uv run --no-sync supermimic record   --checkpoint logs/q1_fresh/checkpoint_5000.pt   --motion-dir datasets/motion_lafan/complex_motion   --output-dir logs/video_5000 --seed 0
```

每 5000 轮自动抽取一整条录像：

```bash
uv run --no-sync supermimic watch-videos   --run-dir logs/q1_fresh   --motion-dir datasets/motion_lafan/complex_motion --interval 5000
```

使用 NVIDIA GPU 无屏绘图时，可为 watcher 添加 `--render-device 0`。

## CUDA 单卡与八卡对照

`configs/cuda-bench.example.toml` 用仓库自带的模型与一条固定参考动作跑 30 轮。环境数和批量是全局总量，两次运行保持一致。先确认 CUDA 能看到 8 张卡，再分别启动：

```bash
nvidia-smi -L
nvidia-smi topo -m
uv sync --python 3.13 --no-dev
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() >= 8'
RUN_TAG=$(date +%Y%m%d-%H%M%S)
for CARDS in 1 8; do
  .venv/bin/supermimic launch --config configs/cuda-bench.example.toml \
    --gpus "$CARDS" --output-dir "logs/a6000-${RUN_TAG}-${CARDS}gpu" || break
done
```

比较两个输出目录里的 `throughput.json` 和 `metrics.jsonl`。短测先验证兼容性和固定工作量下的扩展效率，下一步可为多卡调大总环境数并按同一动作的跟踪效果评估收益。

### A6000 八卡后续短测

`cuda-bench-scale.example.toml` 将全局仿真环境从 1024 增至 4096，保持学习批量不变。
`cuda-bench-graph.example.toml` 保持原短测全局规模，同时启用 Inductor 和想象 rollout 的 CUDA Graph。
两套配置分别与原八卡报告对比。首次编译与捕获的耗时记录在首轮 `update_seconds`，后 15 轮各阶段均值在 `throughput.json`。

### A6000 八卡长训

`configs/a6000-8gpu.example.toml` 针对单机 8 张 A6000。每卡 512 个仿真环境，世界模型和策略批量各 256。MLP 使用 BF16，优化器使用 CUDA fused Adam。策略的 32 步 rollout 与反向传播使用 CUDA Graph，默认关闭 Inductor 编译，避免每个 rank 在起步时编译整套内核。第一次捕获时终端会显示 `Capturing the actor CUDA Graph (eager)` 和 `Actor CUDA Graph ready`。

用下面的命令从零开始，继续复用容器的 CUDA PyTorch 环境：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv-a6000/bin/python -m supermimic launch \
  --config configs/a6000-8gpu.example.toml --gpus 8 \
  --output-dir "logs/a6000-train-$(date +%Y%m%d-%H%M%S)"
```

默认从 Yuanke 动作库选 16 段动作。只训练某一段舞时，把配置复制到同一目录并修改 `motion_dir`、`motion_pattern` 和 `max_motions`。首次运行确认首轮捕获成功，并用后续训练日志中的阶段耗时、仿真步数和显存余量调优。本配置的八卡实测应在目标服务器完成。

### 复用已有 PyTorch 的 CUDA 容器

对于已提供 CUDA 版 PyTorch 的容器（例如 Python 3.12 和 CUDA 12.8 的 NVIDIA 镜像），可以使用容器内的 Python 和 PyTorch，从源码启动。以下安装只写入当前仓库里的虚拟环境。

```bash
python3 -m venv --system-site-packages --without-pip .venv-a6000
python3 -m pip --python "$PWD/.venv-a6000/bin/python" install -r requirements-cuda-container.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv-a6000/bin/python -c 'import torch, numpy as np, mujoco, mujoco_warp, warp; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() >= 8; print(torch.from_numpy(np.zeros(1, dtype=np.float32)).cuda())'
```

使用这个环境跑上面的 CUDA 短测时，将命令中的 `.venv/bin/supermimic launch` 换成 `.venv-a6000/bin/python -m supermimic launch`。

## Apple Silicon 训练

在 M4 Pro / 48 GB Mac 上可用 `configs/mac-m4-pro-mps.example.toml`。MuJoCo、动作采集和经验回放使用 CPU，世界模型、策略和价值网络在 MPS 上更新。启动器的 `--gpus 1` 表示单进程，设备由配置中的 `device = "mps"` 选择。

```sh
uv sync --python 3.13 --no-dev
cp configs/mac-m4-pro-mps.example.toml configs/mac-local.toml
.venv/bin/supermimic launch --config configs/mac-local.toml --gpus 1
```

配置使用 128 个仿真环境和 2048 批量。每次新运行使用新的 `output_dir`。续训时设置 `resume` 为检查点路径，同时保留对应的 `.ranks` 目录，并将 `iterations` 设为累计目标轮数。各阶段耗时和 MPS 驱动内存会在指定轮次写入 `metrics.jsonl`。策略阶段还会记录参考数据准备、想象 rollout、策略反向传播、价值目标计算和价值网络拟合的耗时。Mac 配置启用 `time_major_rollout`，在进入循环前一次性排列参考张量，减少每步复制。

在该机 1024 批量对照中，MPS 每轮采集、模型和策略阶段合计约 1.83 秒，CPU 约 3.24 秒。MPS 在 2048 批量时约 2.65 秒，4096 批量时约 6.04 秒。2048 在这组测试中每秒处理的策略样本最多，驱动报告内存约 11 GiB；4096 约占 22 GiB。小批量或当前进程无法访问 Metal 时，可以使用 `configs/mac-m4-pro.example.toml` 的 CPU 配置。

### 网页动作预览

启动新训练时加 `--webui`。训练已经在跑时，可在另一个终端启动独立预览：

```sh
.venv/bin/python -m supermimic preview --run-dir logs/mac_m4_pro_mps_train
```

打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。左侧播放参考动作，右侧播放策略在 MuJoCo 中的实际表现。底部标记首次失稳时间和跟踪误差。默认使用固定的第一条动作，回放前 6 秒，并随新检查点更新。可用 `--motion-index`、`--seconds` 和 `--port` 调整独立预览。回放需要 FFmpeg，Apple Silicon 使用虚拟环境里的 `mjpython` 渲染。TensorBoard 可单独用 `tensorboard --logdir <运行目录>/tensorboard` 打开。

## 开发

```bash
make check
make test
uv build
```

其他导入、评测和参数入口见 `uv run --no-sync supermimic --help`。
代码采用 [MIT License](LICENSE)，依赖与来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
