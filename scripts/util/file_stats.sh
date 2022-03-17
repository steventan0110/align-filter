LANGUAGE=ps
ROOT=/home/steven/Code/GITHUB/proxy-filter/
SCORE_DIR=${ROOT}/filter.output/ps
DATASET=${ROOT}/dataset/ps/filter # use filtered data (by language id and content overlap)
SRC_DATA=${DATASET}/ps-en.ps
TGT_DATA=${DATASET}/ps-en.en

echo "python ${ROOT}/code/file_stats.py \
  --score-file "${SCORE_DIR}/reg/filter/proxy/scores.txt, ${SCORE_DIR}/laser/scores.txt" \
  --src-file ${SRC_DATA} --tgt-file ${TGT_DATA} \
  --step-size 1000000"


