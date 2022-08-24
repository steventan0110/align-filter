# Bitext Mining for Low Resource Languages via Contrastive Learning

This is a repository that performs sentence alignment and sentence filtering for hierarchical mining. Experiments focus on WMT2020 Corpus Filtering Task, which provided Khmer and Pashto's noisy mining data. Note that the experiments are all performed on a qsub-based cluster environment and a lot of codes are not cleaned up yet. 

## Project Structure
````
-- code:
  -- algin: code that fine-tunes sentence-BERT (or other pre-trained models) with Multiple Negatives Ranking Loss
  -- filter: Modified version of XLM-Roberta fine-tune from HUAWEI's [submission](https://aclanthology.org/2020.wmt-1.105.pdf) to WMT2020
-- scripts: scripts used to perform various experiments for sentence alignment and filtering.
  -- main.sh: entry point (execute this to execute the whole alignment and filtering pipeline)
-- util: scripts for visualization or computing text statistics. Also contains important python files used by scripts folder to perform parallel computation (for sentence alignment)
````

## Citation (Bibtex)
To be added
