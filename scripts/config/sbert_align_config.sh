# config file of using sentence-bert to align documents
export parallel_number=10
export assemble_size=100 # number of embeddings to group together to speedup things
export sbert_model_name='bert-base-multilingual-cased'
export sbert_epochs=1
export window_size=6
export neg_samples=6
export emb_prefix="train_tune_sbert"
export sbert_num_samples=20000
export SBERT_CHECKPOINT_FOLDER=${CHECKPOINT}/align/${lang}/${sbert_model_name}-${sbert_epochs}-${window_size}-${neg_samples}-${sbert_num_samples}
export aligned_dir=${DATASET}/sentence-align/${lang}/sbert
export alignment_type=sbert-align
