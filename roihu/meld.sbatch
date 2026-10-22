#!/bin/bash
# Runs meld-dataset.ipynb on one Roihu GH200 GPU with papermill. Cell output streams into meld-<jobid>.out and
# the executed notebook (tables and figures included) is saved as /scratch/<project>/<user>/meld/meld-dataset_<jobid>.ipynb.
# After `bash roihu/setup_env.sh <project>`, submit from the folder holding meld-dataset.ipynb on roihu-gpu.csc.fi:
#
#   sbatch --account=project_2001234 --time=01:00:00 roihu/meld.sbatch debug        # quick check (MELD_DEBUG=1)
#   sbatch --account=project_2001234 roihu/meld.sbatch                              # full run
#   MELD_FT_MODEL=Qwen/Qwen3-14B sbatch --account=project_2001234 roihu/meld.sbatch  # another fine-tuned text model
#
# Resubmitting continues where a job stopped: MELD, model weights, features and finished runs stay in scratch.
#SBATCH --job-name=meld-moe
#SBATCH --partition=gpumedium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=217086
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=16:00:00
#SBATCH --output=meld-%j.out

module purge
module load python-pytorch/2.13
source /projappl/${SLURM_JOB_ACCOUNT}/${USER}/meld-venv/bin/activate
set -eo pipefail

BASE=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/meld
export MELD_WORK_DIR=${BASE}/work                  # features, runs and results
export MELD_DOWNLOAD_DIR=${BASE}/raw               # MELD.Raw video clips (~11 GB), downloaded once
export HF_HOME=${BASE}/hf-cache                    # model weights (~20 GB with Qwen3-8B), downloaded once
export MELD_FT_MODEL=${MELD_FT_MODEL:-Qwen/Qwen3-8B}
export TOKENIZERS_PARALLELISM=false
if [ "${1:-}" = debug ]; then
    export MELD_DEBUG=1
fi
mkdir -p "$MELD_WORK_DIR" "$MELD_DOWNLOAD_DIR" "$HF_HOME"

OUT=${BASE}/meld-dataset_${SLURM_JOB_ID}${1:+_$1}.ipynb
echo "fine-tuned text model: $MELD_FT_MODEL | executed notebook: $OUT"
srun python -m papermill "${SLURM_SUBMIT_DIR}/meld-dataset.ipynb" "$OUT" -k meld-venv --cwd "$SLURM_SUBMIT_DIR" \
     --log-output --no-progress-bar --request-save-on-cell-execute
