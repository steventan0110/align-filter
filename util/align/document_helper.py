"""" This file is used to retrieve individual documents from WMT2020 file, as well as align them with given alignment"""
from pathlib import Path
import argparse
import base64
import os
import multiprocessing as mp
from multiprocessing import Process, Manager

def save_doc(params):
	cpu_id, content = params
	if len(content) == 0:
		return
	for item in content:
		en, other, en_doc_name, other_doc_name = item
		with open(en_doc_name, 'w') as en_handle, open(other_doc_name, 'w') as other_handle:
			en_handle.write(en)
			other_handle.write(other)
	return

def retrieve_alignment(docs, bucket):
	for item in bucket:
		en_file, other_file, alignment_file = item
		doc = align_doc(en_file, other_file, alignment_file)
		docs.append(doc)


def get_docs(buckets):
	with Manager() as manager:
		docs = manager.list()
		processes = []
		for i in range(args.n_cpu):
			p = Process(target=retrieve_alignment, args=(docs, buckets[i]))
			p.start()
			processes.append(p)
		for p in processes:
			p.join()
		return list(docs)

def main(args):
	if args.mode == "decode":
		bucket_size = args.n_cpu # parallelize the saving jobs
		buckets = [(_, []) for _ in range(bucket_size)]
		# decode the documents and store them into temp folder
		bin_size = 1000 # clean memory per 1000 files
		with open(args.input, 'r') as f:
			data = f.read()
		file_counter = 0
		for i, line in enumerate(data.split('\n')):
			if len(line.strip()) < 1: continue
			if file_counter != 0 and file_counter % bin_size == 0:
				with mp.Pool(args.n_cpu) as mpp:
					mpp.map(save_doc, buckets)
				buckets =  [(_, []) for _ in range(bucket_size)]
			bucket_id = file_counter % bucket_size
			en_url, other_url, en, other = line.split('\t')
			en = base64.b64decode(en).decode('utf-8')
			other = base64.b64decode(other).decode('utf-8')
			if len(en) > 512 or len(other) > 512: continue
			en_doc_name = f"{args.output_dir}/{file_counter}.en"
			other_doc_name = f"{args.output_dir}/{file_counter}.{args.lang}"
			buckets[bucket_id][-1].append((en, other, en_doc_name, other_doc_name))
			file_counter += 1
			if file_counter == 10: break
		with mp.Pool(args.n_cpu) as mpp:
			mpp.map(save_doc, buckets)
	else:
		# assemble the documents and align them according to alignment file from vecalign
		n_files = len([name for name in os.listdir(args.input)])
		size = n_files // 10 # hardcode 10 because that's how many files are their for each idx
		bin_size = 1000 # cleanup text when bin size reached to save memory usage
		bin_count = 0
		bucket_size = args.n_cpu  # parallelize the saving jobs
		buckets = [[] for _ in range(bucket_size)]
		index = 0
		while os.path.exists(os.path.join(args.input, f'{index}.en.overlap')):
			if index != 0 and index % bin_size == 0:
				docs = get_docs(buckets)
				buckets = [[] for _ in range(bucket_size)]
				start_bin, end_bin = bin_count * bin_size, min(size, (bin_count + 1) * bin_size)
				output_file = f"{args.output_dir}/en-{args.lang}.{args.prefix}.{start_bin}-{end_bin}"
				with open(output_file, 'w') as f:
					f.write('\n'.join(docs)+'\n')
				bin_count += 1

			en_file = f"{args.input}/{index}.en"
			other_file = f"{args.input}/{index}.{args.lang}"
			alignment = f"{args.input}/{index}.en-{args.lang}.{args.prefix}"
			bucket_id = index % bucket_size
			buckets[bucket_id].append((en_file, other_file, alignment))
			index += 1

		has_remaining_doc = any([len(bucket) > 0 for bucket in buckets])
		if has_remaining_doc:
			docs = get_docs(buckets)
			buckets = [[] for _ in range(bucket_size)]
			start_bin, end_bin = bin_count * bin_size, min(size, (bin_count + 1) * bin_size)
			output_file = f"{args.output_dir}/en-{args.lang}.{args.prefix}.{start_bin}-{end_bin}"
			with open(output_file, 'w') as f:
				f.write('\n'.join(docs)+'\n')


def align_doc(en, other, align):
	""" align sentences """
	with open(en, 'r') as en_h, open(other, 'r') as other_h, open(align, 'r') as align_h:
		en_data = en_h.read().split('\n')
		other_data = other_h.read().split('\n')
		alignment = align_h.read().split('\n')[:-1]
	output = []
	for line in alignment:
		src_idx, tgt_idx = process_alignment(line)
		src_sent = []
		tgt_sent = []
		for i in src_idx:
			if i == '': break # do not align anything
			src_sent.append(en_data[int(i)])
		for i in tgt_idx:
			if i == '': break # do not align anything
			tgt_sent.append(other_data[int(i)])
		if len(src_sent) == 0 or len(tgt_sent) == 0:
			continue # no alignment for this case
		align_text = ' '.join(src_sent) + '\t' + ' '.join(tgt_sent)
		output.append(align_text)
	return '\n'.join(output)

def process_alignment(input):
	src, tgt = input.split(':')[0], input.split(':')[1]
	src = src[1:-1].split(',') # remove []
	tgt = tgt[1:-1].split(',')
	return src, tgt


def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument('--mode', type=str, choices={'decode', 'assemble'})
	parser.add_argument('--input', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--lang', type=str, default='ps')
	parser.add_argument('--n-cpu', type=int, default=4)
	parser.add_argument('--prefix', type=str, default=None)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	main(args)
