#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -N test-sbert-km
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
source ${CONFIG}/sbert_align_config.sh


LASER_Embed () {
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
SBERT_Embed () {
  ll=$1
  txt=$2
  embed=$3
  # sbert embed
  python ${ROOT}/util/align/sbert_embed.py --input ${txt} --output ${embed} \
    --mode finetune --model-dir ${SBERT_CHECKPOINT_FOLDER}
}


if [[ ! -e ${SBERT_CHECKPOINT_FOLDER} ]]; then
  mkdir -p $SBERT_CHECKPOINT_FOLDER
  echo "finetune the sbert model for alignment"
  python ${ROOT}/code/align/finetune.py \
    --src-data-dir ${DATASET}/wmt/${lang}/train.${lang}-en.${lang} \
    --tgt-data-dir ${DATASET}/wmt/${lang}/train.${lang}-en.en \
    --num-samples ${sbert_num_samples} \
    --checkpoint-dir ${SBERT_CHECKPOINT_FOLDER} --epochs ${sbert_epochs}
else
  echo "SBERT is already finetuned"
fi

# construct pseudo samples if there is no pseudo dataset
output_dir=${DATASET}/pseudo-align/${lang}
if [[ -e ${output_dir} ]]; then
  echo "pseudo dataset already constructed, start evaluation"
else
  mkdir -p ${DATASET}/pseudo-align/${lang}
  python ${ROOT}/util/align/pseudo_align.py --output-dir ${output_dir} \
    --dev-src-dir ${DATASET}/wmt/${lang}/dev.${lang}-en.${lang} \
    --dev-tgt-dir ${DATASET}/wmt/${lang}/dev.${lang}-en.en \
    --train-src-dir ${DATASET}/wmt/${lang}/train.${lang}-en.${lang} \
    --train-tgt-dir ${DATASET}/wmt/${lang}/train.${lang}-en.en \
    --lang ${lang}
fi

# evaluate with finetuned sbert
for folder in ${output_dir}/0*; do
  echo "Evaluting on pseudo aligned dataset ${folder}"
  if [[ ! -e ${folder}/${lang}-en.en.overlap ]]; then
    python ${VECALIGN_DIR}/overlap.py -i ${folder}/${lang}-en.${lang} -o ${folder}/${lang}-en.${lang}.overlap -n 10
    python ${VECALIGN_DIR}/overlap.py -i ${folder}/${lang}-en.en -o ${folder}/${lang}-en.en.overlap -n 10
  fi

  if [[ ! -e ${folder}/${lang}-en.${lang}.overlap.emb.laser ]]; then
    LASER_Embed ${lang} ${folder}/${lang}-en.${lang}.overlap ${folder}/${lang}-en.${lang}.overlap.emb laser
    LASER_Embed en ${folder}/${lang}-en.en.overlap ${folder}/${lang}-en.en.overlap.emb laser
  fi
  if [[ ! -e ${folder}/${lang}-en.${lang}.overlap.emb.sbert ]]; then
    SBERT_Embed ${lang} ${folder}/${lang}-en.${lang}.overlap ${folder}/${lang}-en.${lang}.overlap.emb.sbert
    SBERT_Embed en ${folder}/${lang}-en.en.overlap ${folder}/${lang}-en.en.overlap.emb.sbert
  fi

  result_output_dir=${OUTPUT}/pseudo-align/${lang}
  mkdir -p ${result_output_dir}
  for sbert_type in "sbert" "laser"; do
    echo "VECALIGN Result for ${sbert_type}"
    python ${VECALIGN_DIR}/vecalign.py --alignment_max_size 8 \
      --gold ${folder}/${lang}-en.${lang}en \
      --src ${folder}/${lang}-en.${lang} --tgt ${folder}/${lang}-en.en \
      --src_embed ${folder}/${lang}-en.${lang}.overlap ${folder}/${lang}-en.${lang}.overlap.emb.${sbert_type}  \
      --tgt_embed ${folder}/${lang}-en.en.overlap ${folder}/${lang}-en.en.overlap.emb.${sbert_type} > ${result_output_dir}/alignment.$sbert_type
  done
done



