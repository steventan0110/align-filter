""" visualization of embedding on a small amount of data """
import torch
from torch import nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from laserembeddings import Laser

def load_data(dir):
	with open(dir, 'r') as f:
		return f.read().split('\n')[18:25] # only laod 10 sentence for visualization

class Visualizer:
	def __init__(self, src_data_dir, tgt_data_dir):
		self.src_data_dir = src_data_dir
		self.tgt_data_dir = tgt_data_dir
		self.src_sentence = load_data(self.src_data_dir)
		self.tgt_sentence = load_data(self.tgt_data_dir)
		self.laser_encoder = Laser()
		self.st_encoder = SentenceTransformer('all-MiniLM-L6-v2')
		self.st_finetune_encoder = SentenceTransformer(
			'/home/steven/Code/GITHUB/rl_align/checkpoints/ps_train_finetune/bert-base-multilingual-cased-1-6-6')
		self.plot_sim_matrix()

	def sim_matrix(self, src_emb, tgt_emb):
		sim = np.zeros((len(self.src_sentence), len(self.tgt_sentence)))
		for i in range(len(self.src_sentence)):
			for j in range(len(self.tgt_sentence)):
				sim[i, j] = util.cos_sim(src_emb[i, :], tgt_emb[j, :])
		return sim

	def plot_sim_matrix(self):
		src_embed = self.laser_encoder.embed_sentences(self.src_sentence, 'de')
		tgt_embed = self.laser_encoder.embed_sentences(self.tgt_sentence, 'fr')
		src_embed_st = self.st_encoder.encode(self.src_sentence)
		tgt_embed_st = self.st_encoder.encode(self.tgt_sentence)
		src_embed_st_finetune = self.st_finetune_encoder.encode(self.src_sentence)
		tgt_embed_st_finetune = self.st_finetune_encoder.encode(self.tgt_sentence)

		sim_matrix = self.sim_matrix(src_embed, tgt_embed)
		sim_matrix_st = self.sim_matrix(src_embed_st, tgt_embed_st)
		sim_matrix_st_finetune = self.sim_matrix(src_embed_st_finetune, tgt_embed_st_finetune)

		# add some temperature to sim matrix
		# alpha = 0.7
		# for r in range(sim_matrix_st.shape[0]):
		# 	sim_matrix_st[r, :] = np.power(sim_matrix_st[r, :], alpha)
		# 	sim_matrix_st[r, :] /= sum(sim_matrix_st[r, :])
		sns.heatmap(sim_matrix, annot=True)
		plt.figure()
		sns.heatmap(sim_matrix_st, annot=True)
		plt.figure()
		sns.heatmap(sim_matrix_st_finetune, annot=True)
		plt.show()

if __name__ == '__main__':
	src_data_dir = "/home/steven/Code/GITHUB/rl_align//dataset//ps/docs-finetune-dev/0.en"
	tgt_data_dir = "/home/steven/Code/GITHUB/rl_align//dataset//ps/docs-finetune-dev/0.ps"
	visualizer = Visualizer(src_data_dir, tgt_data_dir)