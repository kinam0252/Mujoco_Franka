#!/bin/bash
#SBATCH --job-name=mj_replay_v2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_replay_v2_%A_%a.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_replay_v2_%A_%a.err

source /home/nas_main/kinamkim/.venvs/mujoco_render/bin/activate
cd /home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka

export LD_LIBRARY_PATH=/home/nas_main/kinamkim/.local/lib/gl:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

DATA_DIR="/home/nas_main/kinamkim/Repos/Intern/assets/lift_data_v2"

mapfile -t ALL_EPS < <(ls -d "$DATA_DIR"/kinam_v2_* | sort)
N_EPS=${#ALL_EPS[@]}
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-4}
PER_TASK=$(( (N_EPS + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))
[[ $END -gt $N_EPS ]] && END=$N_EPS

echo "=== MuJoCo Batch Replay v2 ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, Node: $(hostname)"
echo "Episodes: $START..$((END-1)) of $N_EPS (task $SLURM_ARRAY_TASK_ID/$N_TASKS)"
echo "Time: $(date)"
echo ""

for (( i=START; i<END; i++ )); do
    ep="${ALL_EPS[$i]}"
    echo "--- [$((i-START+1))/$((END-START))] $(basename "$ep") ---"
    python3 -u src/batch_replay.py \
        --data-dir "$DATA_DIR" \
        --single "$ep" \
        --fps 15
done

echo ""
echo "=== Finished: $(date) ==="
