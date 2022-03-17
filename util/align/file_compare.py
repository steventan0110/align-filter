import argparse
from tqdm import tqdm
def parser_args():
	parser = argparse.ArgumentParser(description='Use Laser score file to output the bilingual dataset')
	parser.add_argument('--f1', required=True, default='.', type=str)
	parser.add_argument('--f2', required=True, default='.', type=str)
	args = parser.parse_args()
	return args

def main(args):
	with open(args.f1, 'r', errors='ignore') as f1:
		f1_data = f1.read()
	with open(args.f2, 'r', errors='ignore') as f2:
		f2_data = f2.read()
	# compute coverage of same text content
	file1 = f1_data.split('\n')[:-1]
	file2 = f2_data.split('\n')[:-1]
	number_same = 0
	total = len(file2)
	for item in tqdm(file2):
		if item in file1:
			number_same += 1
	print("Number of file2 sent in file1: {}, total length of file2: {}".format(number_same, total))

	print("Percentage of file2 in file1 is {0}".format(number_same/total))


if __name__ == '__main__':
	args = parser_args()
	main(args)