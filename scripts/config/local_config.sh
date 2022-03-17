# config file for running code locally
export ROOT=/home/steven/Code/GITHUB/align-filter/
export THIRD_PARTY=/home/steven/Code/GITHUB/exp
export DATASET=${ROOT}/dataset
export DATABIN=${ROOT}/data-bin
export CHECKPOINT=${ROOT}/checkpoints
export SCRIPT=${ROOT}/scripts
export CONFIG=${SCRIPT}/config
export VECALIGN_DIR=${THIRD_PARTY}/vecalign
export BPE_TOKENS=5000
export BPEROOT=/home/steven/Code/GITHUB/subword-nmt/subword_nmt
export LASER=${THIRD_PARTY}/LASER
export LASER_SCORING=${LASER}/laser-scoring
export laser_model_dir=${LASER}/models
export laser_encoder="${laser_model_dir}/bilstm.93langs.2018-12-26.pt"
export laser_bpe_codes="${laser_model_dir}/93langs.fcodes"
export moses_scripts=${THIRD_PARTY}/mosesdecoder/scripts
export tokenizer=$moses_scripts/tokenizer/tokenizer.perl
export clean=$moses_scripts/training/clean-corpus-n.perl
export norm_punc=$moses_scripts/tokenizer/normalize-punctuation.perl
export rem_non_print_char=$moses_scripts/tokenizer/remove-non-printing-char.perl