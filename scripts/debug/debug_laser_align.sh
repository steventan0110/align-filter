# debug the issue that LASER sometimes give CUDA error for certain files
if [[ ${HOME} == '/home/wtan12' ]]; then
  echo "execute clsp environment"
  source /home/gqin2/scripts/acquire-gpu
  source ~/miniconda3/etc/profile.d/conda.sh
  source ${HOME}/align-filter/scripts/config/clsp_config.sh
else
  echo "execute home environment"
  source ~/anaconda3/etc/profile.d/conda.sh
  source ${HOME}/Code/GITHUB/align-filter/scripts/config/local_config.sh
fi

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
for id in 65 86 131 132 377; do
laser_file=${DATASET}/sentence-align/ps/laser/bin-${id}.ps.overlap
Embed ps ${laser_file} ${laser_file}.emb laser
done
