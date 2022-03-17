import torch
import torch.nn as nn
import random
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from transformers import XLMRobertaForSequenceClassification
from over_utils import DataProcessor, InputExample, InputFeatures
random.seed(42)

def file2list(file_name, as_float=False):
    with open(file_name, encoding='utf8', errors='ignore') as f:
        line_list = f.readlines()
    if as_float:
        return [float(l.strip()) for l in line_list]
    else:
        return [l.strip() for l in line_list]


def list2file(l, path):
    with open(path, 'w') as f:
        for item in l:
            f.write("%s\n" % str(item).strip())


def bitext_to_sts(src, trg, pos_ratio=1, rand_ratio= 6, fuzzy_max=60, fuzzy_ratio=0, neigbour_mix=True,
                  training_examples=10000, laser_file=None, proxy_file=None, sent_file=None):
    size = len(src)
    sts = []
    t = {k: v for k, v in enumerate(trg)}
    score_data = None
    if laser_file is not None:
        # load laser score for regression task
        with open(laser_file, 'r') as f:
            score_data = f.read()
    elif proxy_file is not None:
        with open(proxy_file, 'r') as f:
            score_data = f.read()
    elif sent_file is not None:
        with open(sent_file, 'r') as f:
            score_data = f.read()

    if score_data is not None:
        score_map = {}
        for line in score_data.split('\n'):
            if len(line) < 1: continue
            score, en, other = line.split('\t')
            score_map[(en.strip(), other.strip())] = score

    count_found_pair = 0
    count_zero_score = 0
    for i in range(size):
        if i % 1000 == 0: print(f'Processed {i} lines')
        move_on = False
        for j in range(pos_ratio):
            if score_data is not None: # use additional scoring file to label
                if (trg[i].strip(), src[i].strip()) in score_map:
                    score = score_map[(trg[i].strip(), src[i].strip())]
                    count_found_pair += 1
                else:
                    score = 1
                if float(score) == 0: # ignore the worst examples to save training data
                    count_zero_score += 1
                    move_on = True
                    break
                sts.append(src[i].strip() + "\t" + trg[i].strip() + "\t" + str(score))
            else:
                sts.append(src[i].strip() + "\t" + trg[i].strip()+ "\t" + "1.0")
        if move_on:
            continue
        for k in range(rand_ratio):
            sts.append(src[random.randrange(1,size)].strip() + "\t" + trg[i].strip() + "\t" + "0.0")
        if fuzzy_ratio>0:
            matches = process.extract(trg[i], t, scorer=fuzz.token_sort_ratio, limit=25)
            m_index = [m[2] for m in matches if m[1]<fuzzy_max][:fuzzy_ratio]
            for m in m_index:
                sts.append(src[i].strip() + "\t" + trg[m].strip() + "\t" + "0.0")
        
        if neigbour_mix and i<size-2:
            sts.append(src[i].strip() + "\t" + trg[i+1].strip()+ "\t" + "0.0")
            sts.append(src[i].strip() + "\t" + trg[i-1].strip()+ "\t" + "0.0")
    # filter in sts step so that we don't need to convert all of them to features
    print(f'{count_found_pair} sentence pairs are found in scoring file (if given any), '
          f'out of which {count_zero_score} lines have 0 score')
    subsample_sts = random.sample(sts, k=int(int(training_examples)*(pos_ratio+rand_ratio+fuzzy_ratio+2)))
    return subsample_sts


class NMTTXLMRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = NMTTRobertaClassificationHead(config)


class NMTTRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 2048)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(2048, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XSTSProcessor(DataProcessor):
    """Processor for the XSTS data set (Huawei version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def create_train_examples(self, pair_dirs, negative_random_sampling, positive_oversampling,
                              two_way_neighbour_sampling, fuzzy_ratio, fuzzy_max, training_examples,
                              laser_file, proxy_file, sent_file, mode):
        """Creates examples for the training and dev sets."""
        src_lines = []
        trg_lines = []
        for pair in pair_dirs:
            src_lines += file2list(pair["src"])
            trg_lines += file2list(pair["trg"])
        train_sts = bitext_to_sts(src_lines, trg_lines, rand_ratio= negative_random_sampling,
                                  pos_ratio=positive_oversampling, neigbour_mix=two_way_neighbour_sampling,
                                  fuzzy_ratio=fuzzy_ratio, fuzzy_max=fuzzy_max,
                                  training_examples=training_examples, laser_file=laser_file,
                                  proxy_file=proxy_file, sent_file=sent_file)
        examples = []
        for (i, line) in enumerate(train_sts):
            guid = "%s-%s" % ("train", str(i))
            text_a ,text_b, label = line.split("\t")
            if mode == "classification":
                example_label = str(int(float(label)))
            else:
                example_label = str(float(label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=example_label))
        return examples

    def create_valid_examples(self, valid_pair):
        """Creates examples for the training and dev sets."""
        src_lines = file2list(valid_pair["src"])
        trg_lines = file2list(valid_pair["trg"])
        valid_sts = bitext_to_sts(src_lines, trg_lines, rand_ratio=0, pos_ratio=1, fuzzy_ratio=3, neigbour_mix=True)
        examples = []
        for (i, line) in enumerate(valid_sts):
            guid = "%s-%s" % ("dev", str(i))
            text_a ,text_b, label = line.split("\t")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(int(float(label)))))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(int(float(label)))))
        return examples
