# Pretrain script that is supposed to run on clsp only
# needs to finish all sentence-alignment and sentence filtering step before running this code

. ./local-settings.sh

src=$1
src_dim=$2
tgt=$3
tgt_dim=$4

resources='hostname=c*,mem_free=20G,ram_free=20G,gpu=1'
qsub -l $resources -cwd \
	-N vecmap-$src$src_dim-$tgt$tgt_dim -j y \
	-o $ROOT/qsub/vecmap-$src$src_dim-$tgt$tgt_dim.out \
	vecmap.sh $src $src_dim $tgt $tgt_dim