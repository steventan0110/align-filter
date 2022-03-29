# debug the issue that LASER sometimes give CUDA error for certain files
lang=km
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

python ${SCRIPT}/debug/redo_laser.py --out-file ${HOME}/qsub/km-LASER-align-LASER-filter.out --lang en > temp.txt
#python ${SCRIPT}/debug/redo_laser.py --out-file /home/steven/Code/GITHUB/align-filter/scripts/debug/km-LASER-align-LASER-filter.out --lang en > temp.txt
readarray -t a < temp.txt
for en_id in "${a[@]}"; do
  echo "processing $en_id-bin en"
  laser_file=${DATASET}/sentence-align/${lang}/laser/bin-${en_id}.en.overlap
  Embed en ${laser_file} ${laser_file}.emb laser
done


python ${SCRIPT}/debug/redo_laser.py --out-file ${HOME}/qsub/km-LASER-align-LASER-filter.out --lang ${lang} > temp.txt
#python ${SCRIPT}/debug/redo_laser.py --out-file /home/steven/Code/GITHUB/align-filter/scripts/debug/km-LASER-align-LASER-filter.out --lang ${lang} > temp.txt
readarray -t b < temp.txt
for other_id in "${b[@]}"; do
  echo "processing $other_id-bin ${lang}"
  laser_file=${DATASET}/sentence-align/${lang}/laser/bin-${other_id}.${lang}.overlap
  Embed ${lang} ${laser_file} ${laser_file}.emb laser
done

