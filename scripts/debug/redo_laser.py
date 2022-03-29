import argparse

def main(args):
	with open(args.out_file, 'r') as f:
		data = f.read()
	en_broken_bins = []
	other_broken_bins = []
	en_all_bin = []
	other_all_bin = []
	count = 0
	bin_count = 0
	bin_idx = -1
	is_en=True
	for line in data.split('\n'):
		if line.startswith(' - Encoder: bpe to'):
			bin_count += 1
			bin_idx = int(line.split(' ')[-1].split('.')[0].split('-')[1])
			lang = line.split(' ')[-1].split('.')[1]
			if lang == 'en':
				is_en = True
				en_all_bin.append(bin_idx)
			else:
				is_en = False
				other_all_bin.append(bin_idx)
		elif 'cuDNN error' in line:
			count += 1
			if is_en:
				en_broken_bins.append(bin_idx)
			else:
				other_broken_bins.append(bin_idx)
	# find bins ignored by cuda error
	max_id = 3739
	for i in range(max_id):
		if i not in en_all_bin:
			count += 1
			en_broken_bins.append(i)
		if i not in other_all_bin:
			count += 1
			other_broken_bins.append(i)

	#print(f'{count}/{bin_count} bins are broken')
	if args.lang == 'en':
		for t in en_broken_bins:
			print(t)
	else:
		for t in other_broken_bins:
			print(t)



def parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--out-file', type=str)
	parser.add_argument('--lang', type=str)
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parser_args()
	main(args)
