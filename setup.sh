set -e

# Make sure any install is inside a conda or virtual env
if [[ -z "$CONDA_DEFAULT_ENV"  || "$CONDA_DEFAULT_ENV" == "base" ]]; then
    export PIP_REQUIRE_VIRTUALENV=true
fi

# Ensure requirements are up to date
pip install -r requirements.txt
