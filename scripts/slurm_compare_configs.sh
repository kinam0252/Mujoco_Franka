#!/bin/bash
#SBATCH --job-name=compare_cfg
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/compare_cfg_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/compare_cfg_%j.err

# ── Compare ACT & GR00T configs on same cube position ──
# set -e  # disabled: allow individual runs to fail

source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false

# lerobot source for ACT policy imports
export PYTHONPATH=/home/nas_main/kinamkim/Repos/Intern/lerobot_minho/src:${PYTHONPATH:-}

# Fake CUDA for transformers
FAKE_CUDA=/tmp/fake_cuda_$$
mkdir -p $FAKE_CUDA/bin $FAKE_CUDA/lib64 $FAKE_CUDA/include
cat > $FAKE_CUDA/bin/nvcc << 'NVCC'
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 12.8, V12.8.93"
NVCC
chmod +x $FAKE_CUDA/bin/nvcc
export CUDA_HOME=$FAKE_CUDA
export PATH=$FAKE_CUDA/bin:$PATH

cd /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka

OUTDIR="output/compare_all"
mkdir -p "$OUTDIR"

# Cube position from first episode (kinam_20260401_150941_305)
CUBE_POS="0.4296 -0.0801 0.02"
CUBE_YAW="-2.3"

echo "=== Config Comparison Inference ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Cube:   pos=[$CUBE_POS] yaw=${CUBE_YAW}°"
echo "Time:   $(date)"
echo ""

TRAIN_BASE=~/DATA/INTERN/training

# ── ACT configs ──
echo "=========================================="
echo "  ACT Inference"
echo "=========================================="

declare -A ACT_CONFIGS
ACT_CONFIGS["act_224_c20_100k"]="$TRAIN_BASE/act_224_c20/checkpoints/100000/pretrained_model"
ACT_CONFIGS["act_640_c20_100k"]="$TRAIN_BASE/act_640_c20/checkpoints/100000/pretrained_model"
ACT_CONFIGS["act_lerobot_c100_30k"]="$TRAIN_BASE/act_lerobot/checkpoints/030000/pretrained_model"
ACT_CONFIGS["act_lerobot224_c100_30k"]="$TRAIN_BASE/act_lerobot_224/checkpoints/030000/pretrained_model"

for NAME in "${!ACT_CONFIGS[@]}"; do
    CKPT="${ACT_CONFIGS[$NAME]}"
    OUT="$OUTDIR/${NAME}.mp4"
    echo ""
    echo "--- $NAME ---"
    echo "  Checkpoint: $CKPT"
    echo "  Output: $OUT"
    if [ ! -d "$CKPT" ]; then
        echo "  SKIP: checkpoint not found"
        continue
    fi
    python3 -u src/infer_act_mujoco.py \
        --checkpoint "$CKPT" \
        --cube-pos $CUBE_POS \
        --cube-yaw $CUBE_YAW \
        --action-horizon 10 \
        --ema-alpha 0.7 \
        --no-gripper-ema \
        --max-steps 300 \
        --scene-xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
        --output "$OUT" \
        && echo "  ✓ Done" || echo "  ✗ FAILED"
done

# ── GR00T configs ──
echo ""
echo "=========================================="
echo "  GR00T Inference"
echo "=========================================="

declare -A GROOT_CONFIGS
GROOT_CONFIGS["groot_sim_30k"]="$TRAIN_BASE/gr00t_groot_v2_30k_backup/checkpoint-30000"
GROOT_CONFIGS["groot_sim_100k"]="$TRAIN_BASE/gr00t_groot_v2/checkpoint-100000"

# Add real checkpoint if exists
if [ -d "$TRAIN_BASE/gr00t_real_v2/checkpoint-30000" ]; then
    GROOT_CONFIGS["groot_real_30k"]="$TRAIN_BASE/gr00t_real_v2/checkpoint-30000"
fi

for NAME in "${!GROOT_CONFIGS[@]}"; do
    CKPT="${GROOT_CONFIGS[$NAME]}"
    OUT="$OUTDIR/${NAME}.mp4"
    echo ""
    echo "--- $NAME ---"
    echo "  Checkpoint: $CKPT"
    echo "  Output: $OUT"
    if [ ! -d "$CKPT" ]; then
        echo "  SKIP: checkpoint not found"
        continue
    fi
    python3 -u src/infer_gr00t_mujoco.py \
        --checkpoint "$CKPT" \
        --cube-pos $CUBE_POS \
        --cube-yaw $CUBE_YAW \
        --open-loop-horizon 16 \
        --ema-alpha 0.0 \
        --max-steps 300 \
        --scene-xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
        --output "$OUT" \
        && echo "  ✓ Done" || echo "  ✗ FAILED"
done

echo ""
echo "=========================================="
echo "  Results"
echo "=========================================="
echo "Output directory: $(pwd)/$OUTDIR"
ls -lh "$OUTDIR/"*.mp4 2>/dev/null || echo "(no videos found)"
echo ""
echo "=== Finished: $(date) ==="
