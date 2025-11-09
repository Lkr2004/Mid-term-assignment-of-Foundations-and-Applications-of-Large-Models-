#!/bin/bash
# ============================
# Transformer  训练脚本
# ============================

# ---- 基本设置 ----
EPOCHS=10
BATCH_SIZE=64
SEED=42
SAVE_DIR="/root/autodl-tmp/large-model/src/checkpoints/PE16"
RESULT_DIR="/root/autodl-tmp/large-model/results/PE16"
MAX_SAMPLES_TOKENIZER=50000

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --epochs) EPOCHS="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
  esac
  shift
done

# ---- 日志文件 ----
LOG_DIR="/root/autodl-tmp/large-model/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ---- 输出提示 ----
echo "🚀 启动训练 Transformer 模型"
echo "📦 日志路径: $LOG_FILE"
echo "📁 模型保存路径: $SAVE_DIR"
echo "📊 结果保存路径: $RESULT_DIR"
echo "🔒 随机种子: $SEED"
echo "-----------------------------------------"
echo "🖥️  系统环境信息:"
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "  - CUDA Version: $(nvcc --version | grep release | awk '{print $6}')"
echo "  - PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  - Python: $(python3 --version)"
echo "-----------------------------------------"


# ---- 启动训练 ----
python3 /root/autodl-tmp/large-model/src/train16.py \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --save_dir $SAVE_DIR \
  --result_dir $RESULT_DIR \
  --max_samples_tokenizer $MAX_SAMPLES_TOKENIZER \
  --seed $SEED \
  2>&1 | tee $LOG_FILE

# ---- 训练完成提示 ----
echo "✅ 训练结束，日志保存在：$LOG_FILE"
echo "📊 训练结果保存在：$RESULT_DIR"
echo "-----------------------------------------"
