set -eux

su vscode
conda create -y -n mlops-for-llms-workshop python==3.11
conda init
source /home/vscode/.bashrc
conda activate mlops-for-llms-workshop
pip install --no-input -r requirements.txt
