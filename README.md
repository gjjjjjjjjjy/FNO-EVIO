# FNO-EVIO
# FNO-EVIO

FNO-EVIO 是一个事件相机（Event Camera）+ IMU 的 VIO（Visual-Inertial Odometry）研究/工程化代码库。本仓库包含训练入口、评估入口、以及与旧版 fno-FAST 脚本兼容的启动脚本，便于在多数据集（multi_root）与跨标定（cross-calib）场景下复现实验。

## 目录结构（核心）

- `train_fno_vio.py`：训练入口（重构版，支持 YAML + CLI 覆盖）
- `fno_evio/`：核心包
  - `app.py`：组装 dataloader / model / train / eval 的主流程
  - `data/`：数据集与序列切窗（SequenceDataset / CollateSequence）
  - `models/`：模型封装（HybridVIONet）
  - `training/`：训练 loop / step / loss 组合
  - `eval/`：评估逻辑（ATE / RPE 等）
  - `config/`：配置 schema、YAML 加载与 CLI 覆盖
  - `legacy/`：从旧实现迁移/对齐的兼容实现（用于复现实验与对照）
- `scripts/`
  - `run_train.sh`：交互式训练启动脚本（输出写入 `outputs-fnoevio/`）
  - `run_test_cross_calib.sh`：交互式跨标定推理脚本（输出写入 `outputs-fnoevio/`）
- `configs/`：训练/实验 YAML 配置样例
- `tests/`：单元测试（基础回归测试）

## 环境依赖

- Python 3.9+（建议 3.10/3.11）
- PyTorch（支持 CPU/CUDA/MPS）
- 可选依赖：
  - `PyYAML`：读取 `--config` 与 `--calib_yaml`
  - `h5py`：MVSEC 数据评估
  - 其他依赖以你环境中 `pip freeze`/项目实际安装为准

## 快速开始

### 1) 训练（推荐：YAML 驱动）

```bash
cd /Users/gjy/eventlearning/code/FNO-EVIO
python train_fno_vio.py --config configs/train.yaml
```

你也可以用 CLI 覆盖 YAML（只覆盖你显式传入的参数，不会被默认值污染）：

```bash
python train_fno_vio.py --config configs/train.yaml --device cuda --batch_size 2
```

### 2) 用交互式脚本训练（兼容旧工作流）

脚本默认会把结果写到 `outputs-fnoevio/`，避免和旧 `outputs/` 冲突。

```bash
cd /Users/gjy/eventlearning/code/FNO-EVIO
bash scripts/run_train.sh
```

### 3) 评估（TUM-VIE 风格）

```bash
python test_fno_vio.py \
  --dataset_root /path/to/dataset_root \
  --calib_yaml /path/to/calib.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --dt 0.00833 \
  --output_dir /path/to/out
```

如果 `calib_yaml` 含 `multi_root`，可对所有 root 分别评估并汇总：

```bash
python test_fno_vio.py \
  --calib_yaml /path/to/calib.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --eval_all_roots \
  --output_dir /path/to/out
```

### 4) 评估（MVSEC）

```bash
python test_mvsec.py \
  --dataset_root /path/to/MVSEC \
  --sequence indoor_flying1 \
  --checkpoint /path/to/checkpoint.pth \
  --dt 0.2 \
  --rpe_dt 0.5 \
  --output_dir /path/to/out
```

（兼容入口）也可以用：

```bash
python test_mvsec_clean.py ...
```

### 5) 跨标定推理（交互式）

```bash
bash scripts/run_test_cross_calib.sh
```

脚本会：
- 从测试标定 YAML 里读取 `multi_root`
- 对每个数据集分别跑推理并写入独立子目录
- 生成 `summary_results.txt`

## 推荐的目录布局

为了让 `calib yaml` 里的相对路径（如 `dataset/...`）在脚本中自动解析，推荐:
slam-event/
FNO-EVIO/
dataset/

## 常用参数提示

- 多数据集训练：`--multi_root ...` 或 `--multi_calib_yaml ...`
- 逐 root 验证：训练侧 `--eval_all_roots`；评估侧 `test_fno_vio.py --eval_all_roots`
- 输出目录：训练 `--output_dir ...`；脚本默认 `outputs-fnoevio/...`
- 记录训练指标 CSV：训练 `--metrics_csv /path/to/metrics.csv`

## 许可证与引用

- License：待补充
- Citation：待补充