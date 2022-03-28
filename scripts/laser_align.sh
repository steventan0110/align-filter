#!/usr/bin/env bash
PREPARE_DOC=false
COMPUTE_EMBED=false
DIVIDE_EMBED=false
RETRIEVE_ALIGN=false
CONCAT_DEDUP=false

waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

Embed () {
  ll=$1
  txt=$2
  embed=$3
  prefix=$4
  # laser embed
  #rm ${embed}.${prefix}
  cat ${txt} | python3 ${LASER}/source/embed.py \
    --encoder ${laser_encoder} \
    --token-lang ${ll} \
    --bpe-codes ${laser_bpe_codes} \
    --output ${embed}.${prefix} \
    --verbose
}

# retrieve every single document and align them with vecalign
temp_dir=${aligned_dir}
#rm -r ${temp_dir}
if [[ -e ${temp_dir} ]]; then
  echo " already performed laser align "
  exit
fi
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
  echo "---------------Embedding Assembled -------------------------"
  # prepare laser embedding, sequential execution cuz LASER does not support multi-gpu
  for doc in ${temp_dir}/bin-*.en.overlap; do
    Embed en $doc $doc.emb ${emb_prefix}
  done
  echo "------------------LASER & SBERT EMBED Finished for en -----------------------"

  for doc in ${temp_dir}/bin-*.${lang}.overlap; do
    Embed ${lang} ${doc} ${doc}.emb ${emb_prefix}
  done
  wait
  echo "------------------LASER & SBERT EMBED Finished for other -----------------------"
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
  for align_file in ${temp_dir}/en-${lang}.laser.*; do
    cat ${align_file} >> ${temp_dir}/en-${lang}.${emb_prefix}
  done

  # for ps and kh, there is some missing file so we concate then before dedup
  if [[ ${lang} == "ps" || ${lang} == "km" ]]; then
    cat ${DATASET}/document-align/${lang}/missing.en-${lang} >> ${temp_dir}/en-${lang}.${emb_prefix}
  fi
  echo "-----------------Concat Files Finished -----------------------------------"
  # dedup the src-tgt together and use laser score instead of laser mine
  ${THIRD_PARTY}/preprocess/build/bin/dedupe < ${temp_dir}/en-${lang}.${emb_prefix} > ${temp_dir}/train.${lang}-en
#  cat ${temp_dir}/train.${lang}-en | cut -f1 | ${temp_dir}/train.${lang}-en.en
#  cat ${temp_dir}/train.${lang}-en | cut -f2 | ${temp_dir}/train.${lang}-en.${lang}
  echo "-----------------File Deduped, final version Stored ------------------------"
fi
