#!/bin/bash --login
# Runs meld-dataset.ipynb on one Roihu GH200 GPU with papermill. Cell output streams into meld-<jobid>.out and
# the executed notebook (tables and figures included) is saved as /scratch/<project>/<user>/meld/meld-dataset_<jobid>.ipynb.
# After the setup job (roihu/setup_env.sh), submit from the folder holding meld-dataset.ipynb, from any login node
# or the shell of the web interface (www.roihu.csc.fi):
#
#   sbatch --account=project_2001234 --time=01:00:00 roihu/meld.sbatch debug   # quick check (MELD_DEBUG=1)
#   sbatch --account=project_2001234 roihu/meld.sbatch                         # full run, Qwen/Qwen3-8B
#   sbatch --account=project_2001234 roihu/meld.sbatch Qwen/Qwen3-14B          # another fine-tuned text model
#
# Resubmitting continues where a job stopped: MELD, model weights, features and finished runs stay in scratch.
#SBATCH --job-name=meld-moe
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=217086
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=16:00:00
#SBATCH --output=meld-%j.out
#SBATCH --export=HOME,CSC_ENV_INIT_NON_INTERACTIVE=yes

# the GPU nodes are ARM: the environment is built here (login shell + module load), not inherited from the submitting shell
USER=${USER:-$(id -un)}
module purge
module load python-pytorch/2.13
VENV=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/meld-venv     # built by roihu/setup_env.sh
if [ ! -f "$VENV/bin/activate" ]; then
    echo "no environment at $VENV: run roihu/setup_env.sh first"
    exit 1
fi
source "$VENV/bin/activate"
set -eo pipefail

BASE=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/meld
export MELD_WORK_DIR=${BASE}/work                  # features, runs and results
export MELD_DOWNLOAD_DIR=${BASE}/raw               # MELD.Raw video clips (~11 GB), downloaded once
export HF_HOME=${BASE}/hf-cache                    # model weights (~20 GB with Qwen3-8B), downloaded once
export MELD_FT_MODEL=Qwen/Qwen3-8B
export TOKENIZERS_PARALLELISM=false
SUFFIX=
for arg in "$@"; do
    case "$arg" in
        debug) export MELD_DEBUG=1; SUFFIX=_debug ;;
        */*) export MELD_FT_MODEL=$arg ;;
        *) echo "unknown argument '$arg' (expected 'debug' or a Hugging Face model such as Qwen/Qwen3-14B)"; exit 1 ;;
    esac
done
mkdir -p "$MELD_WORK_DIR" "$MELD_DOWNLOAD_DIR" "$HF_HOME"

OUT=${BASE}/meld-dataset_${SLURM_JOB_ID}${SUFFIX}.ipynb
echo "fine-tuned text model: $MELD_FT_MODEL | executed notebook: $OUT"
export SLURM_EXPORT_ENV=ALL
srun python -m papermill "${SLURM_SUBMIT_DIR}/meld-dataset.ipynb" "$OUT" -k meld-venv --cwd "$SLURM_SUBMIT_DIR" \
     --log-output --no-progress-bar --request-save-on-cell-execute
