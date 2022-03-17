# checkpoint_prefix=ha-en-sent-sim-it2
checkpoint_prefix=ps-cls-sbert-tune-align
#for line in 100000 200000 300000 400000 500000 600000 700000; do
#for line in 600000; do
#  tgt=/home/steven/Code/GITHUB/ParaCrawl/checkpoints/${checkpoint_prefix}-${line}/1e-4
#  mkdir -p ${tgt}
#  scp wtan12@login.clsp.jhu.edu:/export/b02/wtan/checkpoints/${checkpoint_prefix}-${line}/1e-4/*best.pt ${tgt}
#done


for line in 7; do
  tgt=/home/steven/Code/GITHUB/proxy-filter/checkpoints/${checkpoint_prefix}-${line}/1e-3
  mkdir -p ${tgt}
  scp wtan12@login.clsp.jhu.edu:/export/b07/wtan12/checkpoints/${checkpoint_prefix}-${line}/1e-3/*best.pt ${tgt}
done