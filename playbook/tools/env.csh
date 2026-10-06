# Server (csh/tcsh) environment. Replace nrasheed with your user and keep every cache off the NFS home.
set user = `whoami`
source ~/efficient-transformers/qeff_env/bin/activate.csh
setenv HF_HOME /local/mnt/workspace/$user/hf
setenv QEFF_HOME /local/mnt/workspace/$user/qeff_cache
setenv QEFF_LOG_PATH /local/mnt/workspace/$user/qeff_logs
setenv HF_HUB_ENABLE_HF_TRANSFER 1
