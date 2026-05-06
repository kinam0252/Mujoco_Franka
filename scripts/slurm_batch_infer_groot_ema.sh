#!/bin/bash
#SBATCH --job-name=gr00t_ema
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/gr00t_ema_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/gr00t_ema_%j.err

# ── GR00T batch inference with EMA smoothing (alpha=0.9) ──

source /home/nas_main/kinamkim/.venvs/groot/bin/activate
cd /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src

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
export TORCH_CUDA_ARCH_LIST="9.0"
export LD_LIBRARY_PATH=/home/nas_main/kinamkim/.local/lib/gl:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "=== GR00T Batch Inference (EMA alpha=0.9) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time:   $(date)"
echo ""

python3 -u batch_infer_gr00t.py \
    --data-dir /home/nas_main/kinamkim/Repos/Intern/assets/lift_data \
    --checkpoint /home/nas_main/kinamkim/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 \
    --task "lift the cube" \
    --max-steps 300 \
    --open-loop-horizon 16 \
    --ema-alpha 0.9 \
    --output-dir /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/output/batch_infer_ema090 \
    --device cuda:0

echo ""
echo "=== Finished: $(date) ==="
