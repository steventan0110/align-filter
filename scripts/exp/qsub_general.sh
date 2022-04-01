lang=$1
alignment_method=$2
filter_method=$3
if [[ ${HOME} == '/home/wtan12' ]]; then
  resources='hostname=c*,mem_free=20G,ram_free=30G'
  now=$(date +"%I-%M-%S")
  job="${lang}-${alignment_method}-align-${filter_method}-filter-${now}"
  qsub \
  	  -l $resources \
  	  -N $job -j y \
  	  -o $HOME/qsub/$job.out \
  	  ${HOME}/align-filter/scripts/main.sh ${alignment_method} ${filter_method} $lang
else
  # execute home environment
  bash ${HOME}/Code/GITHUB/align-filter/scripts/main.sh ${alignment_method} ${filter_method} $lang
fi
