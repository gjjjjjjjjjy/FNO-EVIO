#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE="$(cd "${CODE}/.." && pwd)"

OUT_BASE="${CODE}/outputs"
LOG_BASE="${OUT_BASE}/logs"
mkdir -p "${LOG_BASE}"

DEFAULT_CALIB=""
DEFAULT_CALIB_A=""
DEFAULT_CALIB_B=""
for p in \
    "${CODE}/yaml/mocap-6dof_calib.yaml" \
    "${BASE}/yaml/mocap-6dof_calib.yaml" \
    "${PWD}/yaml/mocap-6dof_calib.yaml" \
    "${PWD}/fno-FAST/yaml/mocap-6dof_calib.yaml"; do
    if [ -f "${p}" ]; then
        DEFAULT_CALIB="${p}"
        break
    fi
done
for p in \
    "${CODE}/yaml/calib-A.yaml" \
    "${BASE}/yaml/calib-A.yaml" \
    "${PWD}/yaml/calib-A.yaml" \
    "${PWD}/fno-FAST/yaml/calib-A.yaml"; do
    if [ -f "${p}" ]; then
        DEFAULT_CALIB_A="${p}"
        break
    fi
done
for p in \
    "${CODE}/yaml/calib-B.yaml" \
    "${BASE}/yaml/calib-B.yaml" \
    "${PWD}/yaml/calib-B.yaml" \
    "${PWD}/fno-FAST/yaml/calib-B.yaml"; do
    if [ -f "${p}" ]; then
        DEFAULT_CALIB_B="${p}"
        break
    fi
done

DEFAULT_CALIB="${DEFAULT_CALIB:-${CODE}/yaml/mocap-6dof_calib.yaml}"
DEFAULT_CALIB_A="${DEFAULT_CALIB_A:-${CODE}/yaml/calib-A.yaml}"
DEFAULT_CALIB_B="${DEFAULT_CALIB_B:-${CODE}/yaml/calib-B.yaml}"

echo "=========================================="
echo "       FNO-EVIO 训练启动脚本"
echo "=========================================="
echo ""

echo "可用 GPU:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || echo "  (无法检测GPU，请手动输入)"
echo ""
read -p "选择 GPU ID (默认 0): " GPU_ID
GPU_ID=${GPU_ID:-0}
echo ">>> 使用 GPU: ${GPU_ID}"
echo ""

echo "选择数据集:"
echo "  1) single - 单数据集 (mocap-6dof)"
echo "  2) multi  - 多数据集 (A+B 联合训练)"
echo "  3) A      - 仅数据集 A (TUM-VIE 标定)"
echo "  4) B      - 仅数据集 B (TUM-VIE 标定)"
read -p "输入选择 [1/2/3/4] (默认 1): " DATASET_CHOICE
case ${DATASET_CHOICE} in
    2|multi)
        DATASET="multi"
        PREFIX="big_AB_"
        DATA_ARGS="--multi_calib_yaml ${DEFAULT_CALIB_A} ${DEFAULT_CALIB_B} --eval_all_roots --balanced_sampling"
        NUM_WORKERS=2
        SEQ_LEN=400
        SEQ_STRIDE=400
        ;;
    3|A|a)
        DATASET="A"
        PREFIX="big_A_"
        DATA_ARGS="--calib_yaml ${DEFAULT_CALIB_A} --eval_all_roots"
        NUM_WORKERS=2
        SEQ_LEN=400
        SEQ_STRIDE=400
        ;;
    4|B|b)
        DATASET="B"
        PREFIX="big_B_"
        DATA_ARGS="--calib_yaml ${DEFAULT_CALIB_B} --eval_all_roots"
        NUM_WORKERS=2
        SEQ_LEN=400
        SEQ_STRIDE=400
        ;;
    *)
        DATASET="single"
        PREFIX=""
        DATA_ARGS="--calib_yaml ${DEFAULT_CALIB}"
        NUM_WORKERS=4
        SEQ_LEN=200
        SEQ_STRIDE=200
        ;;
esac
echo ">>> 数据集: ${DATASET}"
echo ""

echo "选择切窗模式 (A/B):"
echo "  1) imu_time - IMU时间戳等间隔切窗 (默认)"
echo "  2) gt_time  - 旧基线(仅离线，不可部署)"
read -p "输入选择 [1/2] (默认 1): " WIN_MODE
WIN_MODE=${WIN_MODE:-1}
WIN_TAG=""
WINDOWING_ARGS=""
case ${WIN_MODE} in
    2|gt|GT|gt_time|GT_TIME)
        WIN_TAG="_gtWin"
        WINDOWING_ARGS="--windowing_mode gt"
        ;;
    *)
        WIN_TAG="_imuWin"
        WINDOWING_ARGS="--windowing_mode imu"
        ;;
esac
echo ">>> 切窗模式: ${WIN_TAG}"
echo ""

echo "选择训练 Stage:"
echo "  0)  Stage 0  - 最干净 baseline (无约束)"
echo "  1)  Stage 1  - 固定 s=0.5 (sanity check)"
echo "  2)  Stage 2  - MLE(s) 信息矩阵"
echo "  3a) Stage 3a - 仅 static 约束"
echo "  3b) Stage 3b - static + scale 约束 (推荐解决步长漂移)"
echo "  3c) Stage 3c - static + scale + path_scale"
echo "  4)  Stage 4  - Bayesian 不确定性融合 (跨数据集自适应融合)"
echo "  4b) Stage 4b - Bayesian + static/scale/path (推荐跨数据集)"
echo "  4br) Stage 4br - Bayesian 4b复现参考 (独立命名)"
echo "  4bn) Stage 4bn - Bayesian + bias prior (TUM-VIE官方推荐，防止bias漂移)"
echo "  5)  Stage 5  - IMU基准+视觉修正 (推荐大数据集)"
echo "  6)  Stage 6  - Final baseline (全约束，无 Bayesian)"
read -p "输入选择 [0/1/2/3a/3b/3c/4/4b/4br/4bn/5/6] (默认 5): " STAGE
STAGE=${STAGE:-5}

case ${STAGE} in
    0)
        STAGE_NAME="stage0_baseline"
        STAGE_ARGS=""
        ;;
    1)
        STAGE_NAME="stage1_fixed_s"
        STAGE_ARGS="--scale_min 0.5 --scale_max 0.5"
        ;;
    2)
        STAGE_NAME="stage2_mle_s"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0"
        ;;
    3a)
        STAGE_NAME="stage3a_static"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --loss_w_static 0.5 --speed_thresh 0.05"
        ;;
    3b)
        STAGE_NAME="stage3b_static_scale"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --loss_w_static 0.5 --loss_w_scale 0.5 --speed_thresh 0.05"
        ;;
    3c)
        STAGE_NAME="stage3c_full_constraint"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --loss_w_static 0.5 --loss_w_scale 0.5 --loss_w_path_scale 0.1 --speed_thresh 0.05"
        ;;
    4)
        STAGE_NAME="stage4_bayesian"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --uncertainty_fusion --loss_w_uncertainty 0.1 --loss_w_uncertainty_calib 0.05"
        ;;
    4b)
        STAGE_NAME="stage4b_bayes_full"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --uncertainty_fusion --loss_w_uncertainty 0.1 --loss_w_uncertainty_calib 0.05 --loss_w_static 0.5 --loss_w_scale 0.5 --loss_w_path_scale 0.1 --speed_thresh 0.05 --use_seq_scale --seq_scale_reg 0.08 --min_step_threshold 0.001 --min_step_weight 0.1"
        ;;
    4br)
        STAGE_NAME="stage4b_bayes_full_ref"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --uncertainty_fusion --loss_w_uncertainty 0.1 --loss_w_uncertainty_calib 0.05 --loss_w_static 0.5 --loss_w_scale 0.5 --loss_w_path_scale 0.1 --speed_thresh 0.05 --use_seq_scale --seq_scale_reg 0.08 --min_step_threshold 0.001 --min_step_weight 0.1"
        ;;
    4bn)
        STAGE_NAME="stage4bn_bayes_bias_prior"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --uncertainty_fusion --loss_w_uncertainty 0.1 --loss_w_uncertainty_calib 0.05 --loss_w_static 0.5 --loss_w_scale 0.5 --loss_w_path_scale 0.1 --speed_thresh 0.05 --use_seq_scale --seq_scale_reg 0.08 --min_step_threshold 0.001 --min_step_weight 0.1 --loss_w_bias_a 1e-3 --loss_w_bias_g 1e-3"
        ;;
    5)
        STAGE_NAME="stage5_imu_anchored"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --loss_w_correction 0.1 --loss_w_static 0.5 --loss_w_scale 0.3 --loss_w_bias_a 1e-3 --loss_w_bias_g 1e-3 --speed_thresh 0.05"
        ;;
    6)
        STAGE_NAME="stage6_final_baseline"
        STAGE_ARGS="--scale_min 0.0 --scale_max 1.0 --loss_w_static 0.5 --loss_w_scale 0.5 --loss_w_path_scale 0.1 --speed_thresh 0.05 --use_seq_scale --seq_scale_reg 0.08 --min_step_threshold 0.001 --min_step_weight 0.1"
        ;;
    *)
        echo "错误: 未知的 Stage: ${STAGE}"
        exit 1
        ;;
esac
echo ">>> Stage: ${STAGE} (${STAGE_NAME})"
echo ""

echo "消融模式 (会改变训练/验证行为):"
echo "  0) none"
echo "  1) imu_only    - 仅IMU(关闭视觉修正): scale=0 + 强制关闭 Bayesian"
echo "  2) visual_only - 仅视觉(关闭IMU): --no-imu_gate_soft + imu_mask_prob=1.0"
echo "  3) both        - 两者都关(仅用于快速 sanity，不建议做对比)"
read -p "输入选择 [0/1/2/3] (默认 0): " ABL
ABL=${ABL:-0}
ABLATION_ARGS=""
ABL_TAG=""
case ${ABL} in
    1)
        ABL_TAG="_imuOnly"
        ABLATION_ARGS="--scale_min 0.0 --scale_max 0.0 --no-uncertainty_fusion --no-uncertainty_gate --loss_w_uncertainty 0 --loss_w_uncertainty_calib 0"
        ;;
    2)
        ABL_TAG="_visOnly"
        ABLATION_ARGS="--no-imu_gate_soft --imu_mask_prob 1.0"
        ;;
    3)
        ABL_TAG="_bothOff"
        ABLATION_ARGS="--scale_min 0.0 --scale_max 0.0 --no-imu_gate_soft --imu_mask_prob 1.0 --no-uncertainty_fusion --no-uncertainty_gate --loss_w_uncertainty 0 --loss_w_uncertainty_calib 0"
        ;;
    *)
        ABLATION_ARGS=""
        ABL_TAG=""
        ;;
esac

OUT_DIR="${OUT_BASE}/${PREFIX}${STAGE_NAME}${ABL_TAG}${WIN_TAG}"
LOG_FILE="${LOG_BASE}/${PREFIX}${STAGE_NAME}${ABL_TAG}${WIN_TAG}.log"
mkdir -p "${OUT_DIR}"

DT=0.00833
SAMPLE_STRIDE=4
WINDOW_DT=$(awk "BEGIN{printf \"%.6f\", ${DT}}")
if [[ "${WIN_TAG}" = "_imuWin" ]]; then
    WINDOWING_ARGS="${WINDOWING_ARGS} --window_dt ${WINDOW_DT}"
fi

BASE_ARGS="--batch_by_root \
    --dt ${DT} \
    --sample_stride ${SAMPLE_STRIDE} \
    --sequence_len ${SEQ_LEN} \
    --sequence_stride ${SEQ_STRIDE} \
    --window_stack_K 3 \
    --voxel_stack_mode delta \
    --epochs 500 \
    --batch_size 512 \
    --tbptt_len 75 \
    --eval_interval 1 \
    --eval_batch_size 1 \
    --num_workers ${NUM_WORKERS} \
    --persistent_workers \
    --prefetch_factor 1 \
    --seed 42 \
    --imu_gate_soft \
    --mixed_precision \
    --scheduler cosine \
    --warmup_epochs 10 \
    --patience 50 \
    --earlystop_metric ate \
    --no-uncertainty_fusion \
    --no-uncertainty_gate \
    --loss_w_uncertainty 0 \
    --loss_w_uncertainty_calib 0 \
    --loss_w_scale 0 \
    --loss_w_static 0 \
    --loss_w_path_scale 0 \
    --loss_w_scale_reg 0 \
    --no-use_seq_scale \
    --seq_scale_reg 0 \
    --no-adaptive_loss"

echo "=========================================="
echo "训练配置确认:"
echo "  CODE:        ${CODE}"
echo "  GPU:         ${GPU_ID}"
echo "  数据集:      ${DATASET}"
echo "  Stage:       ${STAGE} (${STAGE_NAME})"
echo "  num_workers: ${NUM_WORKERS}"
echo "  seq_len:     ${SEQ_LEN}"
echo "  seq_stride:  ${SEQ_STRIDE}"
echo "  输出目录:    ${OUT_DIR}"
echo "  日志文件:    ${LOG_FILE}"
echo "  切窗:        ${WIN_TAG} (${WINDOWING_ARGS})"
echo "=========================================="
echo ""
read -p "确认启动训练? [y/N]: " CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "启动训练..."

CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u "${CODE}/train_fno_vio.py" \
    ${DATA_ARGS} \
    ${BASE_ARGS} \
    ${WINDOWING_ARGS} \
    ${STAGE_ARGS} \
    ${ABLATION_ARGS} \
    --metrics_csv "${OUT_DIR}/metrics.csv" \
    --output_dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > "${OUT_DIR}/train.pid"

echo ""
echo "=========================================="
echo "训练已启动!"
echo "  PID:  ${PID}"
echo "  停止: kill ${PID}"
echo "  日志: tail -f ${LOG_FILE}"
echo "=========================================="
