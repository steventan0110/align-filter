# use this script to embed and mine the score with laser embedding
filter_method="laser"
deduped_file=${aligned_dir}/train.${lang}-en
filter_dir=${DATASET}/sentence-align-filter/${lang}/${alignment_type}/${filter_method}
src_file=${filter_dir}/train.${lang}-en.${lang}
tgt_file=${filter_dir}/train.${lang}-en.en
if [[ -e "${filter_dir}/${lang}-en.score" ]]; then
  echo "LASER scoring file already computed"
  exit
fi
mkdir -p ${filter_dir}
# first we preprocess the deduped data with overlap and language id check
if true; then
  cat ${deduped_file} | python $LASER_SCORING/filter-stdio.py --overlap 0.9 -l ${lang} -e en > ${deduped_file}.filter
  python ${ROOT}/util/align/preprocess.py --file ${deduped_file}.filter --output-dir ${filter_dir} --lang ${lang}
else
  # debug purpose, realized that using unfiltered data will cuz laser embed error due to sequences that are longer than 512
  cp ${aligned_dir}/train.${lang}-en.${lang} ${filter_dir}/train.${lang}-en.${lang}
  cp ${aligned_dir}/train.${lang}-en.en ${filter_dir}/train.${lang}-en.en
fi

Embed () {
  ll=$1
  txt=$2
  embed=$3
  if [ ! -s ${embed} ] ; then
    cat ${txt} | python3 ${LASER}/source/embed.py \
      --encoder ${laser_encoder} \
      --token-lang ${ll} \
      --bpe-codes ${laser_bpe_codes} \
      --output ${embed} \
      --verbose
  fi
}

Process () {
  Embed en $1.en $1.en.embed
  Embed ${lang} $1.${lang} $1.${lang}.embed

  python3 ${LASER}/source/mine_bitexts.py \
      $1.en $1.${lang} \
      --src-lang en --trg-lang ${lang} \
      --src-embeddings $1.en.embed \
      --trg-embeddings $1.${lang}.embed \
      --mode score --retrieval max --margin ratio -k 4  \
      --output $1.laser --verbose --gpu --unify
}

SPLIT=100000
FILE=$filter_dir
rm -r $FILE.tmp
mkdir -p $FILE.tmp

split -a 5 -d -l $SPLIT ${tgt_file} $FILE.tmp/en.
split -a 5 -d -l $SPLIT ${src_file} $FILE.tmp/${lang}.

# process one part at a time
files=$(ls $FILE.tmp | grep ^en......$)
file_array=($files)
echo "Split into ${#file_array[@]} parts."
for part in $files; do
  id="${part:3:6}"
  mv $FILE.tmp/en.$id $FILE.tmp/$id.en
  mv $FILE.tmp/${lang}.$id $FILE.tmp/$id.${lang}
  if [ ! -s "$FILE.tmp/$id.laser" ] ; then
    Process $FILE.tmp/$id
  fi
done

rm ${FILE}/${lang}-en.score
for scored_file in ${FILE}.tmp/*.laser; do
  cat $scored_file >> ${FILE}/${lang}-en.score
done
rm -r ${FILE}.tmp # clean up

