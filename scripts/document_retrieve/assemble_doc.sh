lang=fr
if [[ ${HOME} == '/home/wtan12' ]]; then
  DATA_ROOT=/home/pkoehn/statmt/data/site-crawl/data/
  SCRIPT=/home/wtan12/align-filter/scripts/document_retrieve/
  OUTPUT_DIR=/export/c07/wtan12/output/assemble_doc/${lang}
else
  DATA_ROOT=/home/steven/Code/GITHUB/align-filter/scripts/document_retrieve/
  SCRIPT=/home/steven/Code/GITHUB/align-filter/scripts/document_retrieve/
  OUTPUT_DIR=/home/steven/Code/GITHUB/align-filter/scripts/document_retrieve/out/$lang
fi
mkdir -p ${OUTPUT_DIR}

python $DATA_ROOT/assemble_doc.py --root ${DATA_ROOT} --lang ${lang} --out ${OUTPUT_DIR}

