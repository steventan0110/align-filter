""" create pseudo alignment using existing aligned data, mostly used as evaluation for finetuned model """
import random
import numpy as np
import argparse
np.random.seed(41)
random.seed(41)
import os
def read_text(dir):
	with open(dir, 'r') as f:
		data = f.read()
	return data.split('\n')

ALIGN=0
GROUP_SRC=1
GROUP_TGT=2
NOISE_SRC=3
NOISE_TGT=4

def main(dev_src_dir, dev_tgt_dir, train_src_dir, train_tgt_dir, output_dir, lang):
	""" create alignment by randomly put some sentences together or inserting some noise """
	dev_src_data = read_text(dev_src_dir)[:200] # only take a portion to save time
	dev_tgt_data = read_text(dev_tgt_dir)[:200]
	train_src_data = read_text(train_src_dir)
	train_tgt_data = read_text(train_tgt_dir)
	src_len = len(dev_src_data)
	tgt_len = len(dev_tgt_data)
	assert src_len == tgt_len
	# perform random action to align the texts and produce corresponding test file for the alignment

	possible_weights = [
		[0.6,0.1,0.1,0.1,0.1], # all combined
		[0.7,0.15,0.15,0,0], # only ins
		[0.7,0,0,0.15,0.15], #only delete
		[0.8,0.2,0,0,0], # ins on src only
		[0.8,0,0.2,0,0], # ins on tgt only
		[0.8,0,0,0.2,0], # delete on src only
		[0.8,0,0,0,0.2] # delete on tgt only
	]
	for assigned_weight in possible_weights:
		align_weight, ins_src_weight, ins_tgt_weight, delete_src_weight, delete_tgt_weight = assigned_weight
		cur_out_dir = "{0}/{1}-{2}-{3}-{4}-{5}".format(output_dir, align_weight, ins_src_weight, ins_tgt_weight,
		                                               delete_src_weight, delete_tgt_weight)
		src_sents = []
		tgt_sents = []
		alignments = []
		src_idx, tgt_idx = 0, 0
		src_align_idx, tgt_align_idx = 0, 0

		if not os.path.exists(cur_out_dir):
			os.mkdir(cur_out_dir)

		while src_idx < src_len and tgt_idx < tgt_len:
			p = int(np.random.choice([ALIGN, GROUP_SRC, GROUP_TGT, NOISE_SRC, NOISE_TGT], size=1, p=assigned_weight))
			# p = random.choices([ALIGN, GROUP_SRC, GROUP_TGT, NOISE_SRC, NOISE_TGT], weights=assigned_weight, k=1)[0]
			# p = random.choices([ALIGN, GROUP_SRC, GROUP_TGT, NOISE_SRC, NOISE_TGT], weights=[5, 1, 1, 1, 1], k=1)[0]
			if p == ALIGN:
				src_sents.append(dev_src_data[src_idx])
				tgt_sents.append(dev_tgt_data[tgt_idx])
				src_idx += 1
				tgt_idx += 1
				alignments.append(((src_align_idx,), (tgt_align_idx,)))
				src_align_idx += 1
				tgt_align_idx += 1
			elif p == GROUP_SRC:
				number_align = int(random.randint(1,4)) # align 1-3 sentence together
				number_align = min(number_align, src_len-src_idx, tgt_len - tgt_idx)
				src_sents.append(' '.join(dev_src_data[src_idx: src_idx+number_align]))
				src_idx += number_align
				tgt_align = []
				for i in range(number_align):
					tgt_sents.append(dev_tgt_data[tgt_idx])
					tgt_idx += 1
					tgt_align.append(tgt_align_idx)
					tgt_align_idx += 1
				alignments.append(((src_align_idx, ), tuple(tgt_align)))
				src_align_idx += 1
			elif p == GROUP_TGT:
				number_align = int(random.randint(1, 4))  # align 1-3 sentence together
				number_align = min(number_align, src_len-src_idx, tgt_len - tgt_idx)
				tgt_sents.append(' '.join(dev_tgt_data[tgt_idx: tgt_idx + number_align]))
				tgt_idx += number_align
				src_align = []
				for i in range(number_align):
					src_sents.append(dev_src_data[src_idx])
					src_idx += 1
					src_align.append(src_align_idx)
					src_align_idx += 1
				alignments.append((tuple(src_align),(tgt_align_idx,)))
				tgt_align_idx += 1
			elif p == NOISE_SRC:
				# add noise from training data
				number_noise = int(random.randint(1, 4))
				src_align = []
				for i in range(number_noise):
					idx = random.randint(0, len(train_src_data))
					src_sents.append(train_src_data[idx])
					src_align.append(src_align_idx)
					src_align_idx += 1
				alignments.append((tuple(src_align), -1)) # -1 indicate nothing is aligned to the noise
			else:
				# add noise from training data
				number_noise = int(random.randint(1, 4))
				tgt_align = []
				for i in range(number_noise):
					idx = random.randint(0, len(train_tgt_data))
					tgt_sents.append(train_tgt_data[idx])
					tgt_align.append(tgt_align_idx)
					tgt_align_idx += 1
				alignments.append((-1, tuple(tgt_align)))  # -1 indicate nothing is aligned to the noise

		# save file and
		src_out = '\n'.join(src_sents) + '\n'
		tgt_out = '\n'.join(tgt_sents) + '\n'
		src_out_file = f'{cur_out_dir}/{lang}-en.{lang}'
		tgt_out_file = f'{cur_out_dir}/{lang}-en.en'
		with open(src_out_file, 'w') as f1:
			f1.write(src_out)
		with open(tgt_out_file, 'w') as f2:
			f2.write(tgt_out)
		write_alignment(cur_out_dir, alignments, lang)


def write_alignment(dir, alignments, lang):
	out = []
	for item in alignments:
		src_idx, tgt_idx = item[0], item[1]
		if src_idx == -1:
			for temp_idx in tgt_idx:
				out_line = f"[]:[{temp_idx}]"
				out.append(out_line)
		elif tgt_idx == -1:
			for temp_idx in src_idx:
				out_line = f"[{temp_idx}]:[]"
				out.append(out_line)
		else:
			out_line = "{0}:{1}".format(list(src_idx), list(tgt_idx))
			out.append(out_line)
	align_out = '\n'.join(out) + '\n'
	align_out_file = f"{dir}/{lang}-en.{lang}en"
	with open(align_out_file, 'w') as f:
		f.write(align_out)

def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument('--dev-src-dir', type=str, required=True)
	parser.add_argument('--dev-tgt-dir', type=str, required=True)
	parser.add_argument('--train-src-dir', type=str, required=True)
	parser.add_argument('--train-tgt-dir', type=str, required=True)
	parser.add_argument('--output-dir', type=str, required=True)
	parser.add_argument('--lang', type=str, default="ps")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	main(args.dev_src_dir, args.dev_tgt_dir, args.train_src_dir, args.train_tgt_dir, args.output_dir, args.lang)