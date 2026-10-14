module load git
export CS_ROOT=/mnt/vstor/courses/csds447/$USER
export HF_HOME=$CS_ROOT/hf_cache
export TORCH_HOME=$CS_ROOT/torch_cache
export PIP_CACHE_DIR=$CS_ROOT/pip_cache
mkdir -p $HF_HOME $TORCH_HOME $PIP_CACHE_DIR
if [ -f $CS_ROOT/venvs/rai/bin/activate ]; then
  source $CS_ROOT/venvs/rai/bin/activate
fi
