#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -N laser-rerun
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M wtan12@jhu.edu
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=20G,gpu=1,hostname=c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

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

python ${SCRIPT}/debug/redo_laser.py --out-file ${HOME}/qsub/km-LASER-align-LASER-filter.out --lang en > ${HOME}/temp.txt
#python ${SCRIPT}/debug/redo_laser.py --out-file /home/steven/Code/GITHUB/align-filter/scripts/debug/km-LASER-align-LASER-filter.out --lang en > temp.txt
readarray -t a < ${HOME}/temp.txt
for en_id in "${a[@]}"; do
  echo "processing $en_id-bin en"
  laser_file=${DATASET}/sentence-align/${lang}/laser/bin-${en_id}.en.overlap
  Embed en ${laser_file} ${laser_file}.emb laser
done


python ${SCRIPT}/debug/redo_laser.py --out-file ${HOME}/qsub/km-LASER-align-LASER-filter.out --lang ${lang} > ${HOME}/temp.txt
#python ${SCRIPT}/debug/redo_laser.py --out-file /home/steven/Code/GITHUB/align-filter/scripts/debug/km-LASER-align-LASER-filter.out --lang ${lang} > temp.txt
readarray -t b < ${HOME}/temp.txt
for other_id in "${b[@]}"; do
  echo "processing $other_id-bin ${lang}"
  laser_file=${DATASET}/sentence-align/${lang}/laser/bin-${other_id}.${lang}.overlap
  Embed ${lang} ${laser_file} ${laser_file}.emb laser
done

