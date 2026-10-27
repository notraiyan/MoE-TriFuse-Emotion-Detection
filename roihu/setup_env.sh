#!/bin/bash --login
# One-time setup on Roihu: a virtual environment on top of CSC's PyTorch module, with the extra packages the
# notebook needs and a Jupyter kernel (meld-venv) for the batch job. It has to be built on the GPU nodes' ARM
# (aarch64) architecture, so submit it as a short job; this works from any login node, including the shell of the
# web interface (www.roihu.csc.fi):
#
#   cd <folder with meld-dataset.ipynb and roihu/>
#   sbatch --account=project_2001234 roihu/setup_env.sh project_2001234      # output: setup-<jobid>.out
#
# On roihu-gpu.csc.fi itself, `bash roihu/setup_env.sh project_2001234` works too.
#SBATCH --job-name=meld-setup
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=217086
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=setup-%j.out
#SBATCH --export=HOME,CSC_ENV_INIT_NON_INTERACTIVE=yes
PROJECT=${1:?usage: sbatch --account=<project> roihu/setup_env.sh <project, e.g. project_2001234>}
USER=${USER:-$(id -un)}
if [ "$(uname -m)" != aarch64 ]; then
    echo "run this as a job (sbatch, see above) or on roihu-gpu.csc.fi; $(hostname) is $(uname -m), not aarch64"
    exit 1
fi
VENV=/scratch/${PROJECT}/${USER}/meld-venv

module purge
module load python-pytorch/2.13
set -eo pipefail
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
# opencv < 5: the notebook's YuNet / Haar face detection was written against OpenCV 4
pip install --no-cache-dir "transformers>=4.56" "peft>=0.13" "opencv-python-headless<5" imageio-ffmpeg ftfy \
    scikit-learn pandas matplotlib papermill ipykernel timm python_speech_features scipy
python -m ipykernel install --prefix "$VENV" --name meld-venv --display-name "MELD (venv)"
python -c "
import torch, transformers, peft, cv2, imageio_ffmpeg
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__,
      '| peft', peft.__version__, '| cv2', cv2.__version__, '| ffmpeg', imageio_ffmpeg.get_ffmpeg_exe())"
echo "environment ready: $VENV"
