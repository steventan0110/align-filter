lang=$1
alignment_method=$2
filter_method=$3
if [[ ${HOME} == '/home/wtan12' ]]; then
  bash ${HOME}/align-filter/scripts/main.sh ${alignment_method} ${filter_method} $lang
else
  # execute home environment
  bash ${HOME}/Code/GITHUB/align-filter/scripts/main.sh ${alignment_method} ${filter_method} $lang
fi
