#!/bin/bash
# MARL训练运行脚本

set -e

# 配置参数
CONFIG_PATH="${1:-config/mappo_example.yaml}"
NUM_GPUS="${2:-2}"

echo "=========================================="
echo "启动MARL训练"
echo "配置文件: ${CONFIG_PATH}"
echo "GPU数量: ${NUM_GPUS}"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "错误: 配置文件 ${CONFIG_PATH} 不存在"
    exit 1
fi

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# 运行训练
python main_marl.py \
    --config "${CONFIG_PATH}" \
    2>&1 | tee marl_training.log

echo "训练完成！日志已保存到 marl_training.log"

