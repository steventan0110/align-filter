alignment_method=$1
filter_method=$2
sample=$3
lang=$4
source /home/gqin2/scripts/acquire-gpu
source /home/wtan12/align-filter/scripts/config/clsp_config.sh
conda activate crawl

output_dir=${CHECKPOINT}/evaluation/${lang}/${alignment_method}/${filter_method}-${sample}
mkdir -p  ${output_dir}
fairseq-train ${DATABIN}/${lang}/${alignment_method}/${filter_method}-${sample} \
  --source-lang ${lang} --target-lang en \
  --arch transformer --share-all-embeddings \
  --encoder-layers 5 --decoder-layers 5 \
  --encoder-embed-dim 512 --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
  --encoder-attention-heads 2 --decoder-attention-heads 2 \
  --encoder-normalize-before --decoder-normalize-before \
  --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
  --weight-decay 0.0001 \
  --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
  --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
  --lr 1e-3 --min-lr 1e-9 \
  --max-tokens 4000 \
  --update-freq 4 \
  --save-dir ${output_dir} \
  --max-epoch 100 --save-interval 10
