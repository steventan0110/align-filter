import copy
from pathlib import Path
import argparse
import json
import os
import multiprocessing as mp
import numpy as np
import shutil

from subprocess import check_output


def wc(filename):
	return int(check_output(["wc", "-l", filename]).split()[0])


def save_embed(filename, embed):
	with open(filename, 'w') as f:
		embed.tofile(f)


def retrieve_embedding(params):
	data_dir, content = params
	if len(content) == 0:
		return  # no workload for this cpu
	for item in content:  # each item corresponds for jobs for one bin
		bin_idx, lang, en_total_len, other_total_len, jobs, prefix = item
		print(f'bin-{bin_idx}.en.overlap.emb.{prefix}')
		en_embed = np.fromfile(
			os.path.join(data_dir, f'bin-{bin_idx}.en.overlap.emb.{prefix}'),
			dtype=np.float32, count=-1)
		# load bin embed which is shared across jobs
		other_embed = np.fromfile(
			os.path.join(data_dir, f'bin-{bin_idx}.{lang}.overlap.emb.{prefix}'),
			dtype=np.float32, count=-1)
		if len(en_embed) == 0 or len(other_embed) == 0:
			# some cuda error occurs for this bin, ignore the embedding
			print("encounter empty embedding file for bin-{0}".format(bin_idx))
			continue

		en_size = en_embed.shape[0] // en_total_len
		en_embed = en_embed.reshape((-1, en_size))
		other_size = other_embed.shape[0] // other_total_len
		other_embed = other_embed.reshape((-1, other_size))

		for job in jobs:
			file_idx, en_length, other_length, en_start_idx, other_start_idx = job
			en_file = f'{file_idx}.en.overlap.emb.{prefix}'
			other_file = f'{file_idx}.{args.lang}.overlap.emb.{prefix}'
			save_embed(os.path.join(data_dir, en_file), en_embed[en_start_idx: (en_start_idx + en_length)])
			save_embed(os.path.join(data_dir, other_file),
			           other_embed[other_start_idx: (other_start_idx + other_length)])


def main(args):
	if args.mode == "assemble":
		# assemble multiple doc's into one file to perform embedding calculation at the same time
		bin_size = args.assemble_size
		bin_number = 0
		cur_idx = 0
		en_file_len = []
		other_file_len = []
		en_files = []
		other_files = []
		bin2len = {}
		while os.path.exists(os.path.join(args.data_dir, f'{cur_idx}.en.overlap')):
			# file assemble not finished yet
			if cur_idx != 0 and cur_idx % bin_size == 0:
				with open(os.path.join(args.data_dir, f'bin-{bin_number}.en.overlap'), 'wb') as wfd:
					for f in en_files:
						with open(f, 'rb') as fd:
							shutil.copyfileobj(fd, wfd)
				with open(os.path.join(args.data_dir, f'bin-{bin_number}.{args.lang}.overlap'), 'wb') as wfdh:
					for h in other_files:
						with open(h, 'rb') as fdh:
							shutil.copyfileobj(fdh, wfdh)
				# save file len stats for later recovery
				bin2len[bin_number] = (en_file_len, other_file_len)
				# new document
				en_files = []
				other_files = []
				bin_number += 1
				en_file_len = []
				other_file_len = []
			en_file_path = os.path.join(args.data_dir, f'{cur_idx}.en.overlap')
			other_file_path = os.path.join(args.data_dir, f'{cur_idx}.{args.lang}.overlap')
			en_files.append(en_file_path)
			other_files.append(other_file_path)
			en_file_len.append(wc(en_file_path))
			other_file_len.append(wc(other_file_path))
			cur_idx += 1
		if len(en_file_len) != 0:
			# has remaining file to process
			with open(os.path.join(args.data_dir, f'bin-{bin_number}.en.overlap'), 'wb') as wfd:
				for f in en_files:
					with open(f, 'rb') as fd:
						shutil.copyfileobj(fd, wfd)
			with open(os.path.join(args.data_dir, f'bin-{bin_number}.{args.lang}.overlap'), 'wb') as wfdh:
				for h in other_files:
					with open(h, 'rb') as fdh:
						shutil.copyfileobj(fdh, wfdh)
			# save file len stats for later recovery
			bin2len[bin_number] = (en_file_len, other_file_len)
		with open(args.length_file, 'w') as fp:
			json.dump(bin2len, fp)
	else:
		# divide embeddings into corresponding index
		bucket_size = args.n_cpu
		buckets = [(args.data_dir, []) for _ in range(bucket_size)]
		bin_size = args.assemble_size
		with open(args.length_file, 'r') as f:
			bin2len = json.load(f)
		for bin_idx in bin2len.keys():
			bin_length = bin2len[bin_idx]
			bin_idx = int(bin_idx)
			en_start_idx, other_start_idx = 0, 0
			en_total_length, other_total_length = sum(bin_length[0]), sum(bin_length[1])
			bucket_id = bin_idx % bucket_size
			work_load = []
			for i in range(len(bin_length[0])):
				file_idx = bin_idx * bin_size + i
				en_length = bin_length[0][i]
				other_length = bin_length[1][i]
				# add files to be processed to buckets for multiprocess

				work_load.append((file_idx, en_length, other_length, en_start_idx, other_start_idx))
				en_start_idx += en_length
				other_start_idx += other_length
			# add workload to corresponding bin-idx
			buckets[bucket_id][-1].append((bin_idx, args.lang,
			                               en_total_length, other_total_length,
			                               work_load, args.prefix))
		with mp.Pool(bucket_size) as mpp:
			mpp.map(retrieve_embedding, buckets)


def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument('--mode', type=str, choices={'divide', 'assemble'})
	parser.add_argument('--length-file', type=Path)
	parser.add_argument('--assemble-size', type=int, default=10)
	parser.add_argument('--data-dir', type=Path, required=True)
	parser.add_argument('--lang', type=str, default='ps')
	parser.add_argument('--n-cpu', type=int, default=4)
	parser.add_argument('--prefix', type=str, default=None, help="prefix for embed name such as laser, sbert, etc.")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	main(args)
