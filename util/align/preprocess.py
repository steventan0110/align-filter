import argparse
from pathlib import Path

def parse_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument('--file', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--lang', type=str, default='ps')
	args = parser.parse_args()
	return args

def main(file, output_dir, lang):
	with open(file, encoding='utf8', errors='ignore') as f:
		lines = f.readlines()
	en_out = []
	other_out = []
	for line in lines:
		if len(line.strip()) < 1: continue
		try:
			en, other = line.strip().split('\t')
		except Exception:
			continue
		if len(en) > 512 or len(other) > 512:
			continue
		en_out.append(en)
		other_out.append(other)
	en_out = '\n'.join(en_out) + '\n'
	other_out = '\n'.join(other_out) + '\n'
	with open(f'{output_dir}/train.{lang}-en.en', 'w') as f1:
		f1.write(en_out)
	with open(f'{output_dir}/train.{lang}-en.{lang}', 'w') as f2:
		f2.write(other_out)


if __name__ == '__main__':
	args = parse_args()
	main(args.file, args.output_dir, args.lang)
	# file = "/home/steven/Code/GITHUB/rl_align/dataset/ps/sbert_finetune/en-ps"
	# output_dir = "/home/steven/Code/GITHUB/rl_align/dataset/ps/sbert_finetune"
	# main(file, output_dir)