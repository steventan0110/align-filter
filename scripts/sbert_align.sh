# main script responsible for aligning the sentence given documents as well as model/embedding to align sentences
# for debug purpose, allow for separate execution
PREPARE_DOC=true
COMPUTE_EMBED=true
DIVIDE_EMBED=true
RETRIEVE_ALIGN=true
CONCAT_DEDUP=true

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

waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

Embed() {
  # sbert embed
  python ${ROOT}/util/align/sbert_embed.py \
    --mode finetune --model-dir ${SBERT_CHECKPOINT_FOLDER}-new --input-dir $1 --prefix $2 --lang $3
}

# retrieve every single document and align them with vecalign
temp_dir=${aligned_dir}
if [[ -e ${temp_dir} ]]; then
  echo " already performed sbert align "
  exit
fi
rm -r ${temp_dir}
mkdir -p ${temp_dir}
if $PREPARE_DOC; then
  python ${ROOT}/util/align/document_helper.py \
    --lang ${lang} --mode decode \
    --input ${DATASET}/document-align/${lang}/docs.en-${lang} \
    --output-dir ${temp_dir} --n-cpu ${parallel_number}
  echo "----------- Documents Stored --------------------------"
  # after creating all temp docs, we align each of them using vecalign
  for doc in ${temp_dir}/*; do
    # prepare overlap file per vecalign's request
    waitforjobs ${parallel_number}
    python ${VECALIGN_DIR}/overlap.py -i ${doc} -o ${doc}.overlap -n 5 &
  done
  wait
  echo "--------------Overlaps Stored ---------------------------"
fi

if $COMPUTE_EMBED; then
  # assemble multiple overlaps into one file for embed
  python ${ROOT}/util/align/embedding_helper.py \
    --lang ${lang} --mode assemble --assemble-size ${assemble_size} \
    --length-file ${temp_dir}/en-${lang}.length.json \
    --data-dir ${temp_dir} --n-cpu ${parallel_number}
  wait
  echo "---------------Embedding Assembled -------------------------"

  Embed ${temp_dir} ${emb_prefix} ${lang}
  wait
  echo "------------------SBERT EMBED Finished-----------------------"
fi

if $DIVIDE_EMBED; then
  # dissemble the embedding
  python ${ROOT}/util/align/embedding_helper.py \
    --lang ${lang} --mode divide --assemble-size ${assemble_size} \
    --length-file ${temp_dir}/en-${lang}.length.json \
    --data-dir ${temp_dir} --n-cpu ${parallel_number} --prefix ${emb_prefix}
  echo "------------------Embedding Dissembled -----------------------------"
fi

if $RETRIEVE_ALIGN; then
  # retrieve alignment
  for doc in ${temp_dir}/*.en; do
    waitforjobs ${parallel_number}
    id=${doc%.*}
    en_file=${id}.en
    en_overlap=${en_file}.overlap
    en_emb=${en_overlap}.emb.${emb_prefix}

    other_file=${id}.${lang}
    other_overlap=${other_file}.overlap
    other_emb=${other_overlap}.emb.${emb_prefix}
    alignment=${id}.en-${lang}

    python ${VECALIGN_DIR}/vecalign.py --alignment_max_size 5  \
      --src ${en_file} --tgt ${other_file} \
      --src_embed ${en_overlap} ${en_emb}  \
      --tgt_embed ${other_overlap} ${other_emb} > ${alignment}.${emb_prefix} &
  done
  wait
  echo "-----------------VECALIGN FINISHED ---------------------------"
fi

if $CONCAT_DEDUP; then
  # assemble the text using alignments
  python ${ROOT}/util/align/document_helper.py \
    --lang ${lang} --mode assemble \
    --input ${temp_dir} \
    --output-dir ${temp_dir} --n-cpu ${parallel_number} --prefix ${emb_prefix}
  echo "----------------------Documents Assembled According to Alignments ---------------"

  # concate all assemble files and dedup
  rm ${temp_dir}/en-${lang}.${emb_prefix}
  for align_file in ${temp_dir}/en-${lang}.${emb_prefix}.*; do
    cat ${align_file} >> ${temp_dir}/en-${lang}.${emb_prefix}
  done
  echo "-----------------Concat Files Finished -----------------------------------"

  # for ps and kh, there is some missing file so we concate then before dedup
  if [[ ${lang} == "ps" || ${lang} == "km" ]]; then
    cat ${DATASET}/document-align/${lang}/missing.en-${lang} >> ${temp_dir}/en-${lang}.${emb_prefix}
  fi
  # dedup the src-tgt together and use laser score instead of laser mine
  ${THIRD_PARTY}/preprocess/build/bin/dedupe < ${temp_dir}/en-${lang}.${emb_prefix} > ${temp_dir}/train.${lang}-en
#  cat ${temp_dir}/train.${lang}-en | cut -f1 > ${temp_dir}/train.${lang}-en.en
#  cat ${temp_dir}/train.${lang}-en | cut -f2 > ${temp_dir}/train.${lang}-en.${lang}
  echo "-----------------File Deduped, final version Stored ------------------------"
fi
