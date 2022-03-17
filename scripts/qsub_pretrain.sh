# Pretrain script that is supposed to run on clsp only
# needs to finish all sentence-alignment and sentence filtering step before running this code
source /home/wtan12/align-filter/scripts/config/clsp_config.sh
lang=ps
resources='hostname=c*,mem_free=20G,ram_free=20G,gpu=1'
for alignment_method in "wmt-align" ; do
  for filter_method in "laser" "sbert" "roberta"; do
  #for filter_method in "laser"; do
    for subsample in 2 3 5 7; do
    #for subsample in 2; do
      qsub -l $resources \
        -N pretrain-${alignment_method}-${filter_method} -j y \
        -o ${HOME}/qsub/pretrain-${alignment_method}-${filter_method}.out \
        ${SCRIPT}/pretrain.sh ${alignment_method} ${filter_method} ${subsample} ${lang}
    done
  done
done



