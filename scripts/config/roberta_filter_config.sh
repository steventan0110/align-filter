export filter_method="roberta"
#export PROXY_TASK_NAME=${alignment_type}
# hard code the roberta model to be laser-align
export PROXY_TASK_NAME="laser-align"
export roberta_max_seq_length=256
export roberta_mode=classification
