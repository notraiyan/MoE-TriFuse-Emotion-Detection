#!/bin/bash
# One-time setup on Roihu: a virtual environment on top of CSC's PyTorch module, with the extra packages the
# notebook needs and a Jupyter kernel (meld-venv) for the batch job. Run it on the GPU login node, which has
# the same ARM architecture as the GPU nodes:
#
#   ssh <user>@roihu-gpu.csc.fi
#   cd <folder with meld-dataset.ipynb and roihu/>
#   bash roihu/setup_env.sh project_2001234
PROJECT=${1:?usage: bash roihu/setup_env.sh <project, e.g. project_2001234>}
if [ "$(uname -m)" != aarch64 ]; then
    echo "run this on roihu-gpu.csc.fi (aarch64), not on $(hostname) ($(uname -m))"
    exit 1
fi
VENV=/projappl/${PROJECT}/${USER}/meld-venv

module purge
module load python-pytorch/2.13
set -eo pipefail
python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
# opencv < 5: the notebook's YuNet / Haar face detection was written against OpenCV 4
pip install --no-cache-dir "transformers>=4.56" "peft>=0.13" "opencv-python-headless<5" imageio-ffmpeg ftfy \
    scikit-learn pandas matplotlib papermill ipykernel
python -m ipykernel install --prefix "$VENV" --name meld-venv --display-name "MELD (venv)"
python -c "
import torch, transformers, peft, cv2, imageio_ffmpeg
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__,
      '| peft', peft.__version__, '| cv2', cv2.__version__, '| ffmpeg', imageio_ffmpeg.get_ffmpeg_exe())"
echo "environment ready: $VENV"
