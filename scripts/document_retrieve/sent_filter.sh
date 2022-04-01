# use to filter the assembled paracrawl sentences
if [[ ${HOME} == '/home/wtan12' ]]; then
  echo "execute clsp environment"
  source ~/miniconda3/etc/profile.d/conda.sh
  source ${HOME}/align-filter/scripts/config/clsp_config.sh
else
  echo "execute home environment"
  source ~/anaconda3/etc/profile.d/conda.sh
  source ${HOME}/Code/GITHUB/align-filter/scripts/config/local_config.sh
fi
lang=ne
assemble_dir=${OUTPUT}/assemble_doc/${lang}
sent_file=${assemble_dir}/en-${lang}
sent_file_dedup=${sent_file}.dedup
# retrieve the sentences, dedup them
if [[ ! -e ${sent_file} ]]; then
  echo "dedup retrieved hunaligned sentences"
  cat ${assemble_dir}/$lang.sent | cut -f3 > ${assemble_dir}/en-$lang.en
  cat ${assemble_dir}/$lang.sent | cut -f4 > ${assemble_dir}/en-$lang.$lang
  paste ${assemble_dir}/en-$lang.en ${assemble_dir}/en-$lang.$lang > ${sent_file}
  ${THIRD_PARTY}/preprocess/build/bin/dedupe < ${sent_file} > ${sentsent_file_dedup_file}
fi

if [[ ! -e ${sent_file_dedup}.filter ]]; then # filter by lang id and coverage
  echo "filter the dedup file with heuristics"
  cat ${sent_file_dedup} | python $LASER_SCORING/filter-stdio.py --overlap 0.9 -l ${lang} -e en > ${sent_file_dedup}.filter
  python ${ROOT}/util/align/preprocess.py --file ${sent_file_dedup}.filter --output-dir ${assemble_dir} --lang ${lang}
fi