# debug the issue that LASER sometimes give CUDA error for certain files
source ~/anaconda3/etc/profile.d/conda.sh
source ${HOME}/Code/GITHUB/align-filter/scripts/config/local_config.sh
conda activate align

Embed () {
  ll=$1
  txt=$2
  embed=$3
  prefix=$4
  # laser embed
  rm ${embed}.${prefix}
  cat ${txt} | python3 ${LASER}/source/embed.py \
    --encoder ${laser_encoder} \
    --token-lang ${ll} \
    --bpe-codes ${laser_bpe_codes} \
    --output ${embed}.${prefix} \
    --verbose
}

laser_file=${SCRIPT}/debug/bin-132.ps.overlap
Embed ps ${laser_file} ${laser_file}.emb laser
