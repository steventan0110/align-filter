import argparse
import os

def main(args):
	# find numbers under root
	all_dir = []
	for _, subdirs, _ in os.walk(args.root):
		for subdir in subdirs:
			print("Processing subdir ", subdir)
			find_related_dirs(os.path.join(args.root, subdir), args.lang, all_dir)
		break
	with open(f'{args.out}/doc.txt', 'w') as f1, open(f'{args.out}/sent.txt', 'w') as f2:
		for temp in all_dir:
			f1.write(f'{temp}/v2.en-{args.lang}.docs.xz\n')
			f2.write(f'{temp}/v2.en-{args.lang}.sent.xz\n')


def find_related_dirs(path, lang, all_dir):
	for _, subdirs, _ in os.walk(path):
		for subdir in subdirs:
			subsubdir = os.path.join(path, subdir)
			if os.path.exists(f'{subsubdir}/v2.en-{lang}.docs.xz') and \
				os.path.exists(f'{subsubdir}/v2.en-{lang}.sent.xz'):
				all_dir.append(subsubdir)

def parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', type=str)
	parser.add_argument('--lang', type=str)
	parser.add_argument('--out', type=str)
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parser_args()
	main(args)