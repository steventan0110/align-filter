""" Use pretrained MLM/Sentence transformer to finetune for classification task"""
import random

from sentence_transformers import InputExample, datasets, models, SentenceTransformer, losses
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm.auto import tqdm  # so we see progress bar
import numpy as np
import argparse

def load_data(src_dir, tgt_dir, score_file):
	src_data = []
	tgt_data = []
	if score_file is not None:
		# passed in optional score file, use it as pre-filter to remove noisy data
		with open(score_file, 'r') as f:
			score_data = f.read().split('\n')
	with open(src_dir, 'r') as f1, open(tgt_dir, 'r') as f2:
		for i, (x, y) in enumerate(zip(f1, f2)):
			if score_file is not None:
				if float(score_data[i]) < 0.9: continue
			x = x.strip()
			y = y.strip()
			if len(x) < 1 or len(y) < 1 or len(x) > 512 or len(y) > 512:
				continue
			src_data.append(x)
			tgt_data.append(y)
	return src_data, tgt_data

def get_data_from_idx(src_sent, tgt_sent, src_idx, tgt_idx, n_neighbor):
	src_out = []
	tgt_out = []
	for src_id in src_idx:
		src_out.append(src_sent[src_id])
	for tgt_id in tgt_idx:
		tgt_out.append(tgt_sent[tgt_id])
	window = n_neighbor // 2
	neighbors = []
	for w in range(1, window+1):
		up = tgt_idx[0]-w
		down = tgt_idx[-1]+w
		if up >= 0:
			neighbors.append(tgt_sent[up])
		if down < len(tgt_sent):
			neighbors.append(tgt_sent[down])

	return ' '.join(src_out), ' '.join(tgt_out), neighbors

def prepare_feature(src_sents, tgt_sents, alignment, window_size, neg_samples):
	# correct alignment is added as premise, neighbors are used as contrastive samples
	data = []
	if alignment is not None:
		for item in alignment:
			src_idx, tgt_idx = item
			if src_idx[0] == -1 or tgt_idx[0] == -1:
				# deletion happens, no usable anchor and positive
				continue
			else:
				src_sent, tgt_sent, neighbors = get_data_from_idx(src_sents, tgt_sents, src_idx, tgt_idx, window_size)
				# use 5 nearby
				for neighbor in neighbors:
					data.append((src_sent, tgt_sent, neighbor))
				# also add some random sample as negative
				neg_counter = 0
				while neg_counter < neg_samples: # use 6 negative samples
					rand_idx = np.random.randint(0, len(tgt_sents), size=1)[0]
					if (tgt_idx[0] - window_size // 2) <= rand_idx <= (tgt_idx[-1] + window_size // 2):
						continue # already sampled by neighbors
					data.append((src_sent, tgt_sent, tgt_sents[rand_idx]))
					neg_counter += 1
	else:
		# assume file is already aligned
		src_idx, tgt_idx = 0, 0
		while src_idx < len(src_sents) and tgt_idx < len(tgt_sents):
			src_sent, tgt_sent, neighbors = get_data_from_idx(src_sents, tgt_sents, [src_idx], [tgt_idx], window_size)
			for neighbor in neighbors:
				data.append((src_sent, tgt_sent, neighbor))
			# also add some random sample as negative
			neg_counter = 0
			while neg_counter < neg_samples:  # use 6 negative samples
				rand_idx = np.random.randint(0, len(tgt_sents), size=1)[0]
				if (tgt_idx - window_size // 2) <= rand_idx <= (tgt_idx + window_size // 2):
					continue  # already sampled by neighbors
				data.append((src_sent, tgt_sent, tgt_sents[rand_idx]))
				neg_counter += 1
			src_idx += 1
			tgt_idx += 1
	return data

def assemble_data_feaure(src_sent, tgt_sent, alignment, window_size, neg_samples, num_samples):
	# first prepare contrastive sentences
	assert len(src_sent) == len(tgt_sent) # same number of files are passed in
	train_samples = []
	for i in range(len(src_sent)):
		cur_alignment = alignment[i] if alignment is not None else None
		data = prepare_feature(src_sent[i], tgt_sent[i], cur_alignment, window_size, neg_samples)
		if len(data) > num_samples:
			data = random.choices(data, k=num_samples)
		# data is structured data with premise and hyp
		for row in tqdm(data):
			train_samples.append(InputExample(
		        texts=[row[0], row[1], row[2]] # row[2] is a hard negative
		    ))
	return train_samples


class BERTFintune:
	def __init__(self):
		pass

class STFinetune:
	def __init__(self, src_data_dir, tgt_data_dir, alignment, checkpoint_dir, epochs, num_samples, score_file=None):
		# init dirs
		# self.src_data_dir = src_data_dir.split(', ')
		# self.tgt_data_dir = tgt_data_dir.split(', ')
		self.alignment = [self.parse_alignment(align_file) for align_file in alignment.split(', ')] \
			if alignment is not None else None
		self.checkpoint_dir = checkpoint_dir
		# init hyperparams and model type
		self.window_size = 6
		self.neg_samples = 6
		self.epochs = epochs
		self.batch_size = 2
		self.num_samples = num_samples
		self.model_card = 'bert-base-multilingual-cased'
		# self.model_card = 'roberta-base'
		# self.output_path = f'{self.checkpoint_dir}/{self.model_card}-{self.epochs}-{self.window_size}-' \
		#                    f'{self.neg_samples}' # checkpoint pass in already has the config
		self.output_path = f'{self.checkpoint_dir}'
		# prepare training features
		# self.src_sent = [load_data(src_dir, score_file) for src_dir in self.src_data_dir]
		# self.tgt_sent = [load_data(tgt_dir, score_file) for tgt_dir in self.tgt_data_dir]
		self.src_sent, self.tgt_sent = load_data(src_data_dir, tgt_data_dir, score_file)
		# TODO: fix the hack [] here
		self.train_samples = assemble_data_feaure([self.src_sent], [self.tgt_sent], self.alignment,
		                                          window_size=self.window_size, 
												  neg_samples=self.neg_samples, num_samples=self.num_samples)
		self.loader = datasets.NoDuplicatesDataLoader(self.train_samples, batch_size=self.batch_size)
		# prepare model
		self.bert = models.Transformer(self.model_card)
		self.pooler = models.Pooling(
			self.bert.get_word_embedding_dimension(),
			pooling_mode_mean_tokens=True
		)
		# wrap model in st
		self.model = SentenceTransformer(modules=[self.bert, self.pooler])
		print(self.model)
		# start finetune
		self.finetune()



	def finetune(self):
		loss = losses.MultipleNegativesRankingLoss(self.model)
		warmup_steps = int(len(self.loader) * self.epochs * 0.1)
		self.model.fit(
			train_objectives=[(self.loader, loss)],
			epochs=self.epochs,
			warmup_steps=warmup_steps,
			output_path=self.output_path,
			show_progress_bar=True
		)

	@staticmethod
	def parse_alignment(alignment):
		def parse_index(input):
			remove_bracket = input[1:-1]  # remove [ ] around the idx
			if len(remove_bracket) == 0:
				# case that no counter increment
				return [-1]
			elif len(remove_bracket.split(', ')) > 1:
				# multiple indexes
				return list(map(lambda x: int(x), remove_bracket.split(', ')))
			else:
				# one index
				return [int(remove_bracket)]

		with open(alignment, 'r') as f:
			alignment_data = f.read()
		out = []
		for i, line in enumerate(alignment_data.split('\n')):
			if len(line) < 1: continue
			left_index = parse_index(line.split(':')[0])
			right_index = parse_index(line.split(':')[1])
			out.append((left_index, right_index))
		return out

def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument('--src-data-dir', type=str, required=True)
	parser.add_argument('--tgt-data-dir', type=str, required=True)
	parser.add_argument('--alignment', type=str, default=None)
	parser.add_argument('--checkpoint-dir', type=str)
	parser.add_argument('--epochs', type=int)
	parser.add_argument('--num-samples', type=int, default=10000)
	parser.add_argument('--score-file', type=str, default=None, help="use score file to filter topk data")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	st_fineune = STFinetune(
		args.src_data_dir,
		args.tgt_data_dir,
		args.alignment,
		args.checkpoint_dir,
		args.epochs,
		args.num_samples,
		args.score_file,
	)
