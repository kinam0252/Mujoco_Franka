#!/bin/bash
#SBATCH --job-name=act_sweep
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/act_sweep_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/act_sweep_%j.err

source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
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

echo "=== ACT EMA Sweep ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Time: $(date)"

python3 -u src/sweep_act_ema.py

echo "=== Finished: $(date) ==="
