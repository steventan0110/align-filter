#!/usr/bin/env bash
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
if [[ ! -e ${src_file} ]]; then
  cat ${deduped_file} | python $LASER_SCORING/filter-stdio.py --overlap 0.9 -l ${lang} -e en > ${deduped_file}.filter
  python ${ROOT}/util/align/preprocess.py --file ${deduped_file}.filter --output-dir ${filter_dir} --lang ${lang}
fi
# check if we have already finetuned xlm-roberta on this dataset
if [[ -e ${CHECKPOINT}/filter/${lang}/${PROXY_TASK_NAME} ]]; then
  echo "Already Finetuned Roberta, skip finetuning step"
else
  mkdir -p ${CHECKPOINT}/cache/cache_data/${lang}
  mkdir -p ${CHECKPOINT}/cache/cache_model/${lang}/
  mkdir -p ${CHECKPOINT}/filter/${lang}/${PROXY_TASK_NAME}
  python ${ROOT}/code/filter/finetune.py \
    --output-mode ${roberta_mode} \
    --model_type xlmroberta \
    --model_name_or_path xlm-roberta-base \
    --task_name $PROXY_TASK_NAME \
    --do_train \
    --logging_steps=5000 \
    --save_steps=20000   \
    --cache_dir ${CHECKPOINT}/cache/cache_model/${lang}/ \
    --examples_cache_dir ${CHECKPOINT}/cache/cache_data/${lang} \
    --max_seq_length ${roberta_max_seq_length} \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_train_batch_size=4 \
    --learning_rate 2e-6 \
    --warmup_steps 1000 \
    --num_train_epochs 2.0 \
    --gradient_accumulation_steps=4 \
    --training_examples 100  \
    --negative_random_sampling 8 \
    --fuzzy_ratio=0 \
    --fuzzy_max_score=60 \
    --positive_oversampling=1   \
    --two_way_neighbour_sampling    \
    --output_dir ${CHECKPOINT}/filter/${lang}/${PROXY_TASK_NAME}  \
    --valid_src_data_dir  ${DATASET}/wmt/${lang}/dev.${lang}-en.${lang}  \
    --valid_trg_data_dir  ${DATASET}/wmt/${lang}/dev.${lang}-en.en \
    --train_src_data_dirs ${filter_dir}/train.${lang}-en.${lang} \
    --train_trg_data_dirs ${filter_dir}/train.${lang}-en.en
fi
wait
python ${ROOT}/code/filter/filter.py \
    --max_seq_length ${roberta_max_seq_length} \
    --output-mode ${roberta_mode} \
    --batch_size 256 \
    --model_checkpoint_path ${CHECKPOINT}/filter/${lang}/${PROXY_TASK_NAME} \
    --src_data ${filter_dir}/train.${lang}-en.${lang}  \
    --trg_data ${filter_dir}/train.${lang}-en.en  \
    --output_dir ${filter_dir}
