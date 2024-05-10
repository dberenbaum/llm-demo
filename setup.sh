set -e

conda create -n mlops-for-llms-workshop python==3.11
conda activate mlops-for-llms-workshop
pip install --no-input -r requirements.txt
