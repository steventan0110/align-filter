import math
import os.path

import numpy as np
import argparse
from pathlib import Path

import tqdm
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, util
import pickle

def parser_args():
	parser = argparse.ArgumentParser(__doc__)
	parser.add_argument('--input', default=None, type=Path)
	parser.add_argument('--output', default=None, type=Path)
	parser.add_argument('--mode', default='pretrain', choices={'pretrain', 'finetune'})
	parser.add_argument('--model-dir', default=None)
	parser.add_argument('--input-dir', default=None)
	parser.add_argument('--prefix', default='laser')
	parser.add_argument('--lang', default='ps')
	return parser.parse_args()

def main(args):
	if args.mode == "pretrain":
		model = SentenceTransformer('all-MiniLM-L6-v2')
		# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
		with open(args.input, 'r') as f:
			input_data = f.read().split('\n')[:-1]

		print('SBERT encode #sentences: ', len(input_data))
		with open(args.output, 'wb') as fout:
			model.encode(input_data).tofile(fout)
	else:
		assert args.model_dir is not None
		model = SentenceTransformer(args.model_dir)
		if args.input_dir is not None:
			cur_idx = 0

			while os.path.exists(os.path.join(args.input_dir, f'bin-{cur_idx}.en.overlap')):
				cur_en_file = os.path.join(args.input_dir, f'bin-{cur_idx}.en.overlap')
				cur_other_file = os.path.join(args.input_dir, f'bin-{cur_idx}.{args.lang}.overlap')
				print('Finetune SBERT for bin = ', cur_idx)
				with open(cur_en_file, 'r', errors='ignore', encoding='utf-8') as f:
					input_data = f.read().split('\n')[:-1]
				output_en = os.path.join(args.input_dir, f'bin-{cur_idx}.en.overlap.emb.{args.prefix}')
				with open(output_en, 'wb') as fout:
					model.encode(input_data).tofile(fout)

				with open(cur_other_file, 'r', errors='ignore', encoding='utf-8') as f1:
					input_data1 = f1.read().split('\n')[:-1]
				output_other = os.path.join(args.input_dir, f'bin-{cur_idx}.{args.lang}.overlap.emb.{args.prefix}')
				with open(output_other, 'wb') as fout1:
					model.encode(input_data1).tofile(fout1)
				cur_idx += 1
		else:
			with open(args.input, 'r') as f:
				input_data = f.read().split('\n')[:-1]

			print('Finetune SBERT encode #sentences: ', len(input_data))
			with open(args.output, 'wb') as fout:
				model.encode(input_data).tofile(fout)

if __name__ == '__main__':
	# debug_laser_online()
	# debug_laser()
	# debug()
	args = parser_args()
	main(args)



def debug():
	cos = nn.CosineSimilarity(dim=0, eps=1e-6)
	#model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
	model = SentenceTransformer('all-MiniLM-L6-v2')
	en1 = "What's your name?"
	en2 = "Happy birthday!"
	en12 = en1 + en2
	zh1 = "Wie heißen Sie?"
	zh2 = "alles Gute zum Geburtstag!"
	zh12 = zh1 + zh2

	en_embed1 = torch.from_numpy(model.encode(en1))
	en_embed2 = torch.from_numpy(model.encode(en2))
	en_embed12 = torch.from_numpy(model.encode(en12))
	zh_embed1 = torch.from_numpy(model.encode(zh1))
	zh_embed2 = torch.from_numpy(model.encode(zh2))
	zh_embed12 = torch.from_numpy(model.encode(zh12))

	en1_en2 = cos(en_embed1, en_embed2)
	en1_en12 = cos(en_embed1, en_embed12)
	zh1_zh2 = cos(zh_embed1, zh_embed2)
	zh1_zh12 = cos(zh_embed1, zh_embed12)

	en1_zh1 = cos(en_embed1, zh_embed1)
	en1_zh2 = cos(en_embed1, zh_embed2)
	en1_zh12 = cos(en_embed1, zh_embed12)

	print("reach here")

def make_norm1(vecs0):
	"""
	make vectors norm==1 so that cosine distance can be computed via dot product
	"""
	for ii in range(vecs0.shape[0]):
		norm = np.sqrt(np.square(vecs0[ii, :]).sum())
		print(norm)
		vecs0[ii, :] = vecs0[ii, :] / (norm + 1e-5)
	return vecs0
def debug_laser():

	cos = nn.CosineSimilarity(dim=0, eps=1e-5)
	data_dir="/home/steven/Code/GITHUB/rl_align/vecalign/toy_data/"
	# temp = np.fromfile(f'{data_dir}/overlaps.en.emb')
	embed_de = np.fromfile(f'{data_dir}/toy.de.emb')
	embed_en = np.fromfile(f'{data_dir}/toy.en2.emb')
	embed_de.resize(2, 512)
	embed_en.resize(2, 512)

	embed_de = make_norm1(embed_de)
	embed_en = make_norm1(embed_en)
	x =torch.from_numpy(embed_de)
	y =torch.from_numpy(embed_en)
	# result = cos(x,y)
	temp = cos(x[0, :], y[0, :])
	temp1 = cos(x[0, :], x[1, :])
	temp2 = cos(x[0, :], y[1, :])
	temp3 = cos(x[1, :], y[0, :])
	temp4 = cos(x[1,:], y[1,:])
	print('r')

def debug_laser_online():
	from laserembeddings import Laser
	model = SentenceTransformer('all-MiniLM-L6-v2')
	laser = Laser()
	de = ["die Grosse der Lichtbrechung , die zu verschiedenen Jahres- und Tageszeiten stark wechselt , die Schwere-Ablenkung durch die Massenanziehung des Himalaya und "]
	# if all sentences are in the same language:
	fr = [
		"a ) la réfraction des rayons lumineux , qui varie fortement selon les saisons et les heures de la journée ;",
		"b ) la déviation des forces d' attraction ( pesanteur ) due au voisinage de la masse de l' Himalaya ;",
		"c ) les réductions au géoïde .",
		"a ) la réfraction des rayons lumineux , qui varie fortement selon les saisons et les heures de la journée ; b ) la déviation des forces d' attraction ( pesanteur ) due au voisinage de la masse de l' Himalaya ; c ) les réductions au géoïde ."
	]
	# de = ["Himalaya-Chronik 1956"]
	# de2 = ["MIT NACHTRÄGEN AUS FRÜHEREN JAHREN VON G.O.DYHRENFURTH"]
	# de3 = ["Mit 3 Bildern "]
	# fr = ["Chronique himalayenne 1956 AVEC NOTES COMPLÉMENTAIRES SUR LES ANNÉES PRÉCÉDENTES PAR G. O. DYHRENFURTH"]
	# de2_embed = laser.embed_sentences(de2, lang='de')
	# de3_embed = laser.embed_sentences(de3, lang='de')
	de_embed = laser.embed_sentences(de, lang='de')  # lang is only used for tokenization
	fr_embed = laser.embed_sentences(fr, lang='fr')
	sbert_de_embed = torch.from_numpy(make_norm1(model.encode(de)))
	sbert_fr_embed = torch.from_numpy(make_norm1(model.encode(fr)))
	x = sbert_de_embed @ sbert_fr_embed.T
	y = np.square(make_norm1(model.encode(de))).sum()
	de_embed = make_norm1(de_embed)
	fr_embed = make_norm1(fr_embed)
	temp = de_embed @ fr_embed.transpose()
	# de2_embed = make_norm1(de2_embed)
	# de3_embed = make_norm1(de3_embed)
	# temp4 = de_embed @ de3_embed.transpose()
	# temp3 = de2_embed @ de3_embed.transpose()
	# temp2 = de_embed @ de2_embed.transpose()
	return




