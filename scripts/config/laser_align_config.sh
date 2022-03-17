# config file of using sentence-bert to align documents
export parallel_number=10
export assemble_size=2 # number of embeddings to group together to speedup things
export model_name='bert-base-multilingual-cased'
export emb_prefix="laser"
export aligned_dir=${DATASET}/sentence-align/${lang}/laser
export alignment_type=laser-align