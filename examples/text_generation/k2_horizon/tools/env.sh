# Laptop (bash) environment for the CPU stages. Adjust ROOT.
ROOT=$HOME/Desktop/Qualcomm-Model-onboarding
source "$ROOT/qeff_env/bin/activate"
export HF_HUB_CACHE="$ROOT/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME="$ROOT/qeff_cache"
# Only if a stored Hugging Face login is expired and breaks public downloads:
# export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
