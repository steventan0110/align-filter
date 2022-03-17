# use this script to embed and mine the score with laser embedding
filter_method="sbert"
deduped_file=${aligned_dir}/train.${lang}-en
filter_dir=${DATASET}/sentence-align-filter/${lang}/${alignment_type}/${filter_method}
src_file=${filter_dir}/train.${lang}-en.${lang}
tgt_file=${filter_dir}/train.${lang}-en.en

mkdir -p ${filter_dir}
# first we preprocess the deduped data with overlap and language id check
if [[ ! -e ${deduped_file}.filter ]]; then
  cat ${deduped_file} | python $LASER_SCORING/filter-stdio.py --overlap 0.9 -l ${lang} -e en > ${deduped_file}.filter
fi
python ${ROOT}/util/align/preprocess.py --file ${deduped_file}.filter --output-dir ${filter_dir} --lang ${lang}
conda activate align # have to use align because of transformer version issue
Embed () {
  ll=$1
  txt=$2
  embed=$3
  # sbert embed
  sbert_model_name='bert-base-multilingual-cased'
  sbert_epochs=1
  window_size=6
  neg_samples=6
  SBERT_CHECKPOINT_FOLDER=${CHECKPOINT}/align/${lang}/${sbert_model_name}-${sbert_epochs}-${window_size}-${neg_samples}
  python ${ROOT}/util/align/sbert_embed.py --input ${txt} --output ${embed} \
    --mode finetune --model-dir ${SBERT_CHECKPOINT_FOLDER}
}

Process () {
  Embed en $1.en $1.en.embed
  Embed ${lang} $1.${lang} $1.${lang}.embed

  python3 ${LASER}/source/mine_bitexts.py \
      $1.en $1.${lang} \
      --src-lang en --trg-lang ${lang} \
      --src-embeddings $1.en.embed \
      --trg-embeddings $1.ps.embed \
      --mode score --retrieval max --margin ratio -k 4  \
      --output $1.$2 --verbose --gpu --unify --dim 768
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
  if [ ! -s "$FILE.tmp/$id.sbert_tune" ] ; then
    Process $FILE.tmp/$id sbert_tune
  fi
done

rm ${FILE}/${lang}-en.score
for scored_file in ${FILE}.tmp/*.sbert_tune; do
  cat $scored_file >> ${FILE}/${lang}-en.score
done
rm -r ${FILE}.tmp # clean up

