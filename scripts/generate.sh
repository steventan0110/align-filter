lang=ps
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

evaluate () {
  alignment_method=$1
  filter_method=$2
  subsample=$3
  mode=$4
  output_dir=${OUTPUT}/evaluation/${lang}/${alignment_method}/${filter_method}-${subsample}
  mkdir -p ${output_dir}
  CHECKPOINT_FOLDER=${CHECKPOINT}/evaluation/${lang}/${alignment_method}/${filter_method}-${subsample}
  DATA_FOLDER=${DATABIN}/${lang}/${alignment_method}/${filter_method}-${subsample}
  filename="transformer"
  fairseq-generate $DATA_FOLDER \
        --task translation \
        --gen-subset $mode \
        --path $CHECKPOINT_FOLDER/checkpoint_best.pt \
        --batch-size 64 \
        --lenpen 1.0 \
        --remove-bpe \
        -s ${lang} -t en \
        --beam 10 > $output_dir/$filename.out

  # detokenize and score
  cat $output_dir/$filename.out | grep ^H | cut -f3- | $moses_scripts/tokenizer/detokenizer.perl > $output_dir/$filename.out.detok
  cat $output_dir/$filename.out | grep ^T | cut -f2- | $moses_scripts/tokenizer/detokenizer.perl > $output_dir/$filename.ref.detok
  # score with sacrebleu
  echo "The BLEU score for ${mode} ${alignment_method}-align ${filter_method}-filter subsample-${subsample} is: "
  sacrebleu $output_dir/$filename.ref.detok -i $output_dir/$filename.out.detok -m bleu -b -w 4
}

conda activate crawl
for alignment_method in "laser-align"; do
  for filter_method in "laser" "sbert" "roberta"; do
  #for filter_method in "roberta"; do
    for subsample in 2 3 5 7; do
      evaluate ${alignment_method} ${filter_method} ${subsample} valid
      wait
      evaluate ${alignment_method} ${filter_method} ${subsample} test
      wait
    done
  done
done


