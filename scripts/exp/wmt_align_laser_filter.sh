lang=ps
if [[ ${HOME} == '/home/wtan12' ]]; then
  resources='hostname=c*,mem_free=20G,ram_free=20G,gpu=1'
  job="wmt-align-laser-filter"
  qsub \
	  -l $resources \
	  -N $job -j y \
	  -o $HOME/qsub/$job.out \
	  ${HOME}/align-filter/scripts/main.sh WMT LASER $lang
  #bash ${HOME}/align-filter/scripts/main.sh WMT LASER $lang
else
  # execute home environment
  bash ${HOME}/Code/GITHUB/align-filter/scripts/main.sh WMT LASER $lang
fi
