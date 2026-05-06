#!/bin/bash
#SBATCH --job-name=mj_render
#SBATCH --partition=core
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_render_%A_%a.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_render_%A_%a.err

# Activate venv
source /home/nas_main/kinamkim/.venvs/mujoco_render/bin/activate

# EGL headless rendering
export LD_LIBRARY_PATH=/home/nas_main/kinamkim/.local/lib/gl:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

DATA_DIR="/home/nas_main/kinamkim/Repos/Intern/assets/lift_data"
SCRIPT_DIR="/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src"

# ── Split episodes across 8 array tasks ──
mapfile -t ALL_EPS < <(ls -d "$DATA_DIR"/kinam_2026* | sort)
N_EPS=${#ALL_EPS[@]}
N_TASKS=8
PER_TASK=$(( (N_EPS + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))
[[ $END -gt $N_EPS ]] && END=$N_EPS

echo "=== MuJoCo Render All (cam_base + cam_wrist) ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Episodes: $START..$((END-1)) of $N_EPS"
echo "Time:   $(date)"
echo ""

for (( i=START; i<END; i++ )); do
    ep="${ALL_EPS[$i]}"
    echo "--- [$((i-START+1))/$((END-START))] $(basename "$ep") ---"
    python3 -u "$SCRIPT_DIR/batch_replay.py" \
        --data-dir "$DATA_DIR" \
        --single "$ep" \
        --fps 15
done

echo ""
echo "=== Finished: $(date) ==="
