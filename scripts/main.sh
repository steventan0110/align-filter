# main script that controls alignment and filter
# options for alignment (either use laser or finetuned sbert)

ALIGN_METHOD=$1
FILTER_METHOD=$2
export lang=$3

echo "Executing ${ALIGN_METHOD} Align with ${FILTER_METHOD} Filtering for ${lang}"

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
if [[ ${ALIGN_METHOD} == "LASER" ]]; then
  source ${CONFIG}/laser_align_config.sh
  bash ${SCRIPT}/laser_align.sh
elif [[ ${ALIGN_METHOD} == "SBERT" ]]; then
  source ${CONFIG}/sbert_align_config.sh
  bash ${SCRIPT}/sbert_align.sh
else
  export aligned_dir=${DATASET}/wmt/${lang}/
  export alignment_type=wmt-align
  echo "do nothing for no alignment option" # this is using wmt released data
fi
exit
conda activate crawl
deduped_file=${aligned_dir}/train.${lang}-en
if [[ ! -e ${deduped_file}.filter ]]; then # filter by lang id and coverage
  cat ${deduped_file} | python $LASER_SCORING/filter-stdio.py --overlap 0.9 -l ${lang} -e en > ${deduped_file}.filter
fi

filter_illegal_sent() {
  method=$1
  filter_dir=${DATASET}/sentence-align-filter/${lang}/${alignment_type}/${method}
  mkdir -p ${filter_dir}
  python ${ROOT}/util/align/preprocess.py --file ${deduped_file}.filter --output-dir ${filter_dir} --lang ${lang}
}

echo "perform sentence filtering on aligned file in ${aligned_dir}"
if [[ ${FILTER_METHOD} == "LASER" ]]; then
  filter_method=laser
  filter_illegal_sent laser
  bash ${SCRIPT}/laser_filter.sh
elif [[ ${FILTER_METHOD} == "SBERT" ]]; then
  filter_method=sbert	
  filter_illegal_sent sbert
  conda activate align # need updatated transformers to run sbert_embed script
  bash ${SCRIPT}/sbert_filter.sh
else # use XLM-Roberta Finetune from HUAWEI's submission to WMT2020, need crawl env because of legacy code
  filter_illegal_sent roberta
  filter_method=roberta
  source ${CONFIG}/roberta_filter_config.sh
  bash ${SCRIPT}/roberta_filter.sh
fi
# generate subsample file and bpe them
bash ${SCRIPT}/filter/prep_score_data.sh ${filter_method}
