#!/bin/bash
filter_method=$1
filter_dir=${DATASET}/sentence-align-filter/${lang}/${alignment_type}/${filter_method}
src_file=${filter_dir}/train.${lang}-en.${lang}
tgt_file=${filter_dir}/train.${lang}-en.en
output_dir=${filter_dir}/subsample
rm -r ${output_dir}
mkdir -p $output_dir
if [ ! -e ${output_dir}/dev.${lang}-en.${lang} ]; then
  # put score together with sentence pairs
  cp ${DATASET}/wmt/${lang}/dev.${lang}-en.${lang} ${output_dir}/dev.${lang}-en.${lang}
  cp ${DATASET}/wmt/${lang}/dev.${lang}-en.en ${output_dir}/dev.${lang}-en.en
  cp ${DATASET}/wmt/${lang}/test.${lang}-en.${lang} ${output_dir}/test.${lang}-en.${lang}
  cp ${DATASET}/wmt/${lang}/test.${lang}-en.en ${output_dir}/test.${lang}-en.en
fi

filter_corpus () {
  lang=$1
  input_file=$2
  output_file=$3

  for threshold in 2 3 5 7; do
    if [ ! -e ${output_file}-${threshold}.en ]; then
      echo "filter corpus with threshold score to certain #lines " $threshold
      python ${ROOT}/code/filter/filter_corpus_with_laser.py \
        --mode word \
        --lang ${lang} --threshold ${threshold} \
        --input ${input_file} --output ${output_file}
    fi
  done
}

tokenize_dataset() {
  lang=$1
  datasets=$2
  for mode in train dev test; do
      for l in $lang en; do
        for threshold in 2 3 5 7; do
          if [ $mode == 'train' ]; then
            cat ${datasets}/${mode}.${lang}-en-${threshold}.$l | \
              perl $norm_punc $l | \
              perl $rem_non_print_char | \
              perl $tokenizer -threads 8 -a -l $l > ${datasets}/tok/${mode}.${lang}-en-${threshold}.$l
          else
            cat ${datasets}/${mode}.${lang}-en.$l | \
              perl $norm_punc $l | \
              perl $rem_non_print_char | \
              perl $tokenizer -threads 8 -a -l $l > ${datasets}/tok/${mode}.${lang}-en.$l
          fi
          done
      done
    done
}

bpe_dataset() {
  lang=$1
  datasets=$2

  for threshold in 2 3 5 7; do
      TRAIN=$datasets/bpe/train.${lang}-en-${threshold}
      rm -f $TRAIN
      for l in ${lang} en; do
          cat ${datasets}/tok/train.${lang}-en-${threshold}.$l >> $TRAIN
      done

      BPECODE=$datasets/bpe/code-${threshold}

      echo "learn BPE code and vocab from training data"
      if [[ ! -e $BPECODE ]]; then
        python $BPEROOT/learn_joint_bpe_and_vocab.py \
          --input ${datasets}/tok/train.${lang}-en-${threshold}.${lang} ${datasets}/tok/train.${lang}-en-${threshold}.en -s ${BPE_TOKENS} -o ${BPECODE} \
          --write-vocabulary ${datasets}/bpe/vocab-${threshold}.${lang} ${datasets}/bpe/vocab-${threshold}.en --num-workers 8
      fi

      for l in ${lang} en; do
          for mode in train dev test; do
              echo "apply_bpe.py to $mode-$threshold.$l"
              if [ $mode == 'train' ]; then
                python $BPEROOT/apply_bpe.py -c $BPECODE --vocabulary ${datasets}/bpe/vocab-${threshold}.${l} \
                --vocabulary-threshold 50 < ${datasets}/tok/${mode}.${lang}-en-${threshold}.$l > $datasets/bpe/${mode}.${lang}-en-${threshold}.$l
              else
                python $BPEROOT/apply_bpe.py -c $BPECODE --vocabulary ${datasets}/bpe/vocab-${threshold}.${l} \
                --vocabulary-threshold 50 < ${datasets}/tok/${mode}.${lang}-en.$l > $datasets/bpe/${mode}.${lang}-en-${threshold}.$l
              fi
          done
      done
    done
}

fairseq_preprocess () {
  lang=$1
  prefix=$2
  datasets=$3
  # apply fairseq preprocess
  for threshold in 2 3 5 7; do
    fairseq-preprocess \
      --source-lang ${lang} --target-lang en \
      --joined-dictionary \
      --trainpref $datasets/bpe/train.${lang}-en-${threshold} \
      --validpref $datasets/bpe/dev.${lang}-en-${threshold} \
      --testpref $datasets/bpe/test.${lang}-en-${threshold} \
      --destdir ${prefix}-${threshold} \
      --workers 8
  done
}
if [ ! -f ${output_dir}/ps-en.score ]; then
  cp ${filter_dir}/ps-en.score ${output_dir}/ps-en.score
fi

filter_corpus ${lang} ${output_dir}/ps-en.score ${output_dir}/train.ps-en

if [ ! -e ${output_dir}/tok ]; then
  mkdir -p ${output_dir}/tok
  tokenize_dataset ${lang} ${output_dir}
fi

if [ ! -e ${output_dir}/bpe ]; then
  mkdir -p ${output_dir}/bpe
  bpe_dataset ${lang} ${output_dir}
fi

fairseq_preprocess ${lang} ${DATABIN}/${lang}/${alignment_type}/${filter_method} ${output_dir}





