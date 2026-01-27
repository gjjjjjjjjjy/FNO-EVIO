#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE="$(cd "${CODE}/.." && pwd)"

OUT_BASE="${BASE}/outputs-fnoevio"
if [ ! -d "${OUT_BASE}" ]; then
    OUT_BASE="${CODE}/outputs-fnoevio"
fi
LOG_BASE="${OUT_BASE}/logs"

DEFAULT_TEST_CALIB=""
for p in \
    "${CODE}/yaml/calib-A.yaml" \
    "${BASE}/yaml/calib-A.yaml" \
    "${PWD}/yaml/calib-A.yaml" \
    "${PWD}/fno-FAST/yaml/calib-A.yaml" \
    "${CODE}/../yaml/calib-A.yaml"; do
    if [ -f "${p}" ]; then
        DEFAULT_TEST_CALIB="${p}"
        break
    fi
done
DEFAULT_TEST_CALIB="${DEFAULT_TEST_CALIB:-${CODE}/yaml/calib-A.yaml}"

DEFAULT_MODEL_CALIB=""
for p in \
    "${CODE}/yaml/calib-B.yaml" \
    "${BASE}/yaml/calib-B.yaml" \
    "${PWD}/yaml/calib-B.yaml" \
    "${PWD}/fno-FAST/yaml/calib-B.yaml" \
    "${CODE}/../yaml/calib-B.yaml"; do
    if [ -f "${p}" ]; then
        DEFAULT_MODEL_CALIB="${p}"
        break
    fi
done
DEFAULT_MODEL_CALIB="${DEFAULT_MODEL_CALIB:-${CODE}/yaml/calib-B.yaml}"

read -p "测试标定 YAML (默认 ${DEFAULT_TEST_CALIB}): " TEST_CALIB
TEST_CALIB=${TEST_CALIB:-${DEFAULT_TEST_CALIB}}
if [[ ! "${TEST_CALIB}" = /* ]]; then
    if [[ "${TEST_CALIB}" = slam-event/* ]]; then
        TEST_CALIB="${TEST_CALIB#slam-event/}"
    fi
    if [ -f "${BASE}/${TEST_CALIB}" ]; then
        TEST_CALIB="${BASE}/${TEST_CALIB}"
    elif [ -f "${CODE}/${TEST_CALIB}" ]; then
        TEST_CALIB="${CODE}/${TEST_CALIB}"
    fi
fi
if [ ! -f "${TEST_CALIB}" ]; then
    echo "错误: 测试标定 YAML 不存在: ${TEST_CALIB}"
    exit 1
fi

echo "(可选) 模型来源标定 YAML，仅用于显示"
read -p "模型标定 YAML (默认 ${DEFAULT_MODEL_CALIB}): " MODEL_CALIB
MODEL_CALIB=${MODEL_CALIB:-${DEFAULT_MODEL_CALIB}}
if [[ ! "${MODEL_CALIB}" = /* ]]; then
    if [[ "${MODEL_CALIB}" = slam-event/* ]]; then
        MODEL_CALIB="${MODEL_CALIB#slam-event/}"
    fi
    if [ -f "${BASE}/${MODEL_CALIB}" ]; then
        MODEL_CALIB="${BASE}/${MODEL_CALIB}"
    elif [ -f "${CODE}/${MODEL_CALIB}" ]; then
        MODEL_CALIB="${CODE}/${MODEL_CALIB}"
    fi
fi

mapfile -t DATASETS < <(python - <<PY
import sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    sys.stderr.write(f"Missing dependency: PyYAML ({e})\n")
    sys.exit(2)

p = Path("${TEST_CALIB}")
obj = yaml.safe_load(p.read_text())
if isinstance(obj, dict) and "value0" in obj and isinstance(obj["value0"], dict):
    obj = obj["value0"]

roots = obj.get("multi_root") if isinstance(obj, dict) else None
if not isinstance(roots, (list, tuple)) or len(roots) == 0:
    sys.stderr.write("No multi_root found in calib YAML. Please add multi_root or use a different YAML.\n")
    sys.exit(3)

for r in roots:
    if r:
        print(str(r))
PY
)

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "错误: 未从 ${TEST_CALIB} 读取到任何数据集路径 (multi_root 为空)"
    exit 1
fi

mkdir -p "${LOG_BASE}"

echo "=========================================="
echo "   FNO-EVIO 跨标定推理脚本"
echo "=========================================="
echo ""
echo "将对以下数据集分别评估:"
for ds in "${DATASETS[@]}"; do
    echo "  - ${ds}"
done
echo ""

echo "可用 GPU:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || echo "  (无法检测GPU，请手动输入)"
echo ""
read -p "选择 GPU ID (默认 0): " GPU_ID
GPU_ID=${GPU_ID:-0}
echo ">>> 使用 GPU: ${GPU_ID}"
echo ""

echo "请输入模型检查点路径:"
read -p "检查点路径: " CHECKPOINT

if [[ ! "${CHECKPOINT}" = /* ]]; then
    if [[ "${CHECKPOINT}" = slam-event/* ]]; then
        CHECKPOINT="${CHECKPOINT#slam-event/}"
    fi
    REL_CP="${CHECKPOINT}"
    BASE_PARENT="$(cd "${BASE}/.." && pwd 2>/dev/null || echo "")"
    CODE_PARENT="$(cd "${CODE}/.." && pwd 2>/dev/null || echo "")"

    CANDIDATES=()
    CANDIDATES+=("${BASE}/${REL_CP}")
    CANDIDATES+=("${CODE}/${REL_CP}")
    if [ -n "${BASE_PARENT}" ]; then
        CANDIDATES+=("${BASE_PARENT}/${REL_CP}")
    fi
    if [ -n "${CODE_PARENT}" ]; then
        CANDIDATES+=("${CODE_PARENT}/${REL_CP}")
    fi
    if [[ "${REL_CP}" = outputs/* ]]; then
        CANDIDATES+=("${OUT_BASE}/${REL_CP#outputs/}")
    fi
    CANDIDATES+=("${OUT_BASE}/${REL_CP}")

    FOUND_CP=""
    for p in "${CANDIDATES[@]}"; do
        if [ -f "${p}" ]; then
            FOUND_CP="${p}"
            break
        fi
    done
    CHECKPOINT="${FOUND_CP:-${BASE}/${REL_CP}}"
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "错误: 检查点文件不存在: ${CHECKPOINT}"
    exit 1
fi
echo ">>> 使用检查点: ${CHECKPOINT}"
echo ""

echo "推理数据格式:"
echo "  1) tum   - TUM-VIE 风格 (test_fno_vio.py)"
echo "  2) mvsec - MVSEC hdf5 (test_mvsec_clean.py)"
read -p "输入选择 [1/2] (默认 1): " TEST_MODE
TEST_MODE=${TEST_MODE:-1}

case ${TEST_MODE} in
    2|mvsec)
        RUN_PREFIX="MVSEC"
        TEST_PY="${CODE}/test_mvsec_clean.py"
        read -p "MVSEC 传感器分辨率 sensor_resolution (H W, 默认 260 346): " SENSOR_RES
        SENSOR_RES=${SENSOR_RES:-"260 346"}
        read -p "MVSEC sequence_stride (默认 50): " MVSEC_SEQ_STRIDE
        MVSEC_SEQ_STRIDE=${MVSEC_SEQ_STRIDE:-50}
        ;;
    *)
        RUN_PREFIX="B2A"
        TEST_PY="${CODE}/test_fno_vio.py"
        ;;
esac
echo ">>> 推理模式: ${RUN_PREFIX}"
echo ""

read -p "评估批次大小 (默认 1): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-1}
echo ""

read -p "时间间隔 dt (秒, 默认 0.00833): " DT
DT=${DT:-0.00833}
echo ""

read -p "RPE 时间间隔 rpe_dt (秒, 默认同 dt=${DT}): " RPE_DT
RPE_DT=${RPE_DT:-${DT}}
echo ""

CKPT_NAME=$(basename "${CHECKPOINT}" .pth)
OUT_DIR_BASE="${OUT_BASE}/test_${RUN_PREFIX}_${CKPT_NAME}"
mkdir -p "${OUT_DIR_BASE}"

echo "=========================================="
echo "推理配置确认:"
echo "  CODE:         ${CODE}"
echo "  GPU:          ${GPU_ID}"
echo "  模型标定:     ${MODEL_CALIB:-'(未设置)'}"
echo "  测试标定:     ${TEST_CALIB}"
echo "  检查点:       ${CHECKPOINT}"
echo "  数据集数量:   ${#DATASETS[@]}"
echo "  dt:           ${DT}s"
echo "  rpe_dt:       ${RPE_DT}s"
echo "  批次大小:     ${BATCH_SIZE}"
echo "  输出目录:     ${OUT_DIR_BASE}"
echo "=========================================="
echo ""
read -p "确认启动推理? [y/N]: " CONFIRM
if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

RESULTS_SUMMARY="${OUT_DIR_BASE}/summary_results.txt"
echo "推理汇总" > "${RESULTS_SUMMARY}"
echo "测试标定: ${TEST_CALIB}" >> "${RESULTS_SUMMARY}"
echo "模型标定: ${MODEL_CALIB:-'(未设置)'}" >> "${RESULTS_SUMMARY}"
echo "检查点: ${CHECKPOINT}" >> "${RESULTS_SUMMARY}"
echo "推理模式: ${RUN_PREFIX}" >> "${RESULTS_SUMMARY}"
echo "dt: ${DT}s" >> "${RESULTS_SUMMARY}"
echo "rpe_dt: ${RPE_DT}s" >> "${RESULTS_SUMMARY}"
echo "日期: $(date)" >> "${RESULTS_SUMMARY}"
echo "" >> "${RESULTS_SUMMARY}"

ALL_SUCCESS=true

for DATASET_PATH in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename "${DATASET_PATH}")

    OUT_DIR="${OUT_DIR_BASE}/${DATASET_NAME}"
    LOG_FILE="${LOG_BASE}/test_${RUN_PREFIX}_${CKPT_NAME}_${DATASET_NAME}.log"
    mkdir -p "${OUT_DIR}"

    if [[ "${DATASET_PATH}" = /* ]]; then
        DATASET_ROOT="${DATASET_PATH}"
    else
        DATASET_ROOT="${BASE}/${DATASET_PATH}"
    fi

    MVSEC_SEQ_ARG=""
    if [[ "${RUN_PREFIX}" = "MVSEC" ]]; then
        if [ ! -d "${DATASET_ROOT}" ]; then
            PARENT_DIR="$(dirname "${DATASET_ROOT}")"
            CAND_DATA="${PARENT_DIR}/${DATASET_NAME}_data.hdf5"
            CAND_GT="${PARENT_DIR}/${DATASET_NAME}_gt.hdf5"
            if [ -d "${PARENT_DIR}" ] && [ -f "${CAND_DATA}" ] && [ -f "${CAND_GT}" ]; then
                DATASET_ROOT="${PARENT_DIR}"
                MVSEC_SEQ_ARG="--sequence ${DATASET_NAME}"
            fi
        fi

        TEST_ARGS="--dataset_root ${DATASET_ROOT} \
            ${MVSEC_SEQ_ARG} \
            --checkpoint ${CHECKPOINT} \
            --dt ${DT} \
            --rpe_dt ${RPE_DT} \
            --batch_size ${BATCH_SIZE} \
            --num_workers 0 \
            --resolution 180 320 \
            --sensor_resolution ${SENSOR_RES} \
            --calib_yaml ${TEST_CALIB} \
            --sequence_stride ${MVSEC_SEQ_STRIDE} \
            --output_dir ${OUT_DIR}"
    else
        TEST_ARGS="--dataset_root ${DATASET_ROOT} \
            --calib_yaml ${TEST_CALIB} \
            --checkpoint ${CHECKPOINT} \
            --dt ${DT} \
            --rpe_dt ${RPE_DT} \
            --batch_size ${BATCH_SIZE} \
            --resolution 180 320 \
            --output_dir ${OUT_DIR}"
    fi

    set +e
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u "${TEST_PY}" \
        ${TEST_ARGS} \
        > "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    set -e

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "----------------------------------------" >> "${RESULTS_SUMMARY}"
        echo "数据集: ${DATASET_NAME}" >> "${RESULTS_SUMMARY}"
        echo "路径: ${DATASET_PATH}" >> "${RESULTS_SUMMARY}"

        if [ -f "${OUT_DIR}/test_results.txt" ]; then
            grep -E "(ATE|RPE_t|RPE_r):" "${OUT_DIR}/test_results.txt" >> "${RESULTS_SUMMARY}" || true
        else
            echo "  [警告] 未找到 ${OUT_DIR}/test_results.txt" >> "${RESULTS_SUMMARY}"
        fi
        echo "" >> "${RESULTS_SUMMARY}"
    else
        echo "----------------------------------------" >> "${RESULTS_SUMMARY}"
        echo "数据集: ${DATASET_NAME}" >> "${RESULTS_SUMMARY}"
        echo "状态: 失败 (退出码: ${EXIT_CODE})" >> "${RESULTS_SUMMARY}"
        echo "" >> "${RESULTS_SUMMARY}"
        ALL_SUCCESS=false
    fi
done

echo "=========================================="
echo "所有推理任务完成!"
echo "汇总结果文件: ${RESULTS_SUMMARY}"
echo "=========================================="
cat "${RESULTS_SUMMARY}"

if [ "$ALL_SUCCESS" = true ]; then
    exit 0
else
    exit 1
fi
