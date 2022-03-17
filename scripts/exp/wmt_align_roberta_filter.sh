lang=ps
if [[ ${HOME} == '/home/wtan12' ]]; then
  bash ${HOME}/align-filter/scripts/main.sh WMT ROBERTA $lang
else
  # execute home environment
  bash ${HOME}/Code/GITHUB/align-filter/scripts/main.sh WMT ROBERTA $lang
fi