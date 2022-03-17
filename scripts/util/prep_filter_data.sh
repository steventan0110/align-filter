#!/usr/bin/env bash

ROOT=/home/steven/Code/GITHUB/proxy-filter
ALIGN_ROOT=/home/steven/Code/GITHUB/rl_align
aligned_text=/home/steven/Code/GITHUB/rl_align/dataset/sbert_finetune2/en-ps.train_tune_sbert
missing_doc=/home/steven/Code/GITHUB/rl_align/dataset/wmt20-sent-missing-in-docs.en-ps
export LASER=/home/steven/Code/GITHUB/rl_align/LASER
export LASER_SCORING=/home/steven/Code/GITHUB/rl_align/LASER/laser-scoring
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"
dataset=$ROOT/dataset/ps/sbert_tune_align2
mkdir -p $dataset

pre_filter_text=$dataset/en-ps.all
cat $aligned_text $missing_doc > $pre_filter_text
# dedup again since we added missing docs
${ALIGN_ROOT}/preprocess/build/bin/dedupe <  $pre_filter_text >  $pre_filter_text.dedup
cat ${pre_filter_text}.dedup | python $LASER_SCORING/filter-stdio.py --overlap 0.9 -l ps -e en > ${pre_filter_text}.filter

python ${ALIGN_ROOT}/util/preprocess.py --file ${pre_filter_text}.filter --output-dir ${dataset}

# check coverage with wmt provided file
released_sent_file=/home/steven/Code/GITHUB/proxy-filter/dataset/ps/wmt20-sent.en-ps
echo "python ${ROOT}/code/file_compare.py --f1 $released_sent_file --f2 ${pre_filter_text}.filter"
