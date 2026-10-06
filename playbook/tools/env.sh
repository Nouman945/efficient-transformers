# bash login on the server. Adjust ROOT to your folder on the local disk.
ROOT=/local/mnt/workspace/$USER
source "$ROOT/efficient-transformers/qeff_env/bin/activate"
export HF_HOME="$ROOT/hf"
export QEFF_HOME="$ROOT/qeff_cache"
export QEFF_LOG_PATH="$ROOT/qeff_logs"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Only if a stored Hugging Face login is expired and breaks public downloads:
# export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
