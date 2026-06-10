#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name="internv-bench1"
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=benchmark1-internvideo-%j.log
#SBATCH --error=benchmark1-internvideo-%j.err
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1

# ======== Environment Setup ========
cd /data/fbau775/InternVideo/InternVideo-Next

source /data/fbau775/miniconda3/bin/activate
conda activate internvideo-next

export HF_HOME=/data/fbau775/.cache/huggingface
export SSL_CERT_FILE=$CONDA_PREFIX/ssl/cacert.pem
export CURL_CA_BUNDLE=$CONDA_PREFIX/ssl/cacert.pem
export MPLCONFIGDIR=/data/fbau775/tmp/matplotlib

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ======== Abort early if no GPU or missing deps ========
python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected — aborting'" || exit 1
python -c "import av" || { echo "ERROR: PyAV not installed (pip install av)"; exit 1; }

# ======== Training + Evaluation for multiple seeds ========
SEEDS=(42)

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Training with seed=${SEED}"
    echo "========================================"
    python train_benchmark1_internvideo.py train \
        --data_dir /data/fbau775/mammalps-dataset/benchmark_1 \
        --model_size base \
        --num_epochs 150 \
        --batch_size 16 \
        --learning_rate 1e-5 \
        --min_learning_rate 1e-7 \
        --weight_decay 0.01 \
        --dropout 0.1 \
        --head_hidden_dim 256 \
        --loss_weight_actions 2.0 \
        --loss_weight_species 1.0 \
        --loss_weight_activity 2.5 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_internvideo_seed_${SEED}" \
        --ckpt_every 50 \
        --keep_recent 5 \
        --output_dir "results/benchmark1_internvideo/train_seed_${SEED}" \
        --seed "$SEED"

    echo ""
    echo "========================================"
    echo "  Evaluating (single-pass) with seed=${SEED}"
    echo "========================================"
    python train_benchmark1_internvideo.py test \
        --data_dir /data/fbau775/mammalps-dataset/benchmark_1 \
        --model_size base \
        --batch_size 16 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_internvideo_seed_${SEED}" \
        --output_dir "results/benchmark1_internvideo/test_seed_${SEED}" \
        --seed "$SEED"

    echo ""
    echo "========================================"
    echo "  Evaluating (multi-sample) with seed=${SEED}"
    echo "========================================"
    python train_benchmark1_internvideo.py test_ms \
        --data_dir /data/fbau775/mammalps-dataset/benchmark_1 \
        --model_size base \
        --head_hidden_dim 256 \
        --batch_size 16 \
        --num_workers 4 \
        --ckpt_dir "checkpoints/benchmark1_internvideo_seed_${SEED}" \
        --output_dir "results/benchmark1_internvideo/test_ms_seed_${SEED}" \
        --num_test_samples 10 \
        --min_clip_duration 0.5 \
        --seed "$SEED"
done

echo ""
echo "========================================"
echo "  All seeds complete"
echo "========================================"
echo "Job finished at $(date)"
