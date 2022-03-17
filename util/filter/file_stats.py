import math
import os.path

import numpy
import matplotlib.pyplot as plt
import argparse
import string
from collections import defaultdict
from pathlib import Path
import tqdm
import torch
from torch import nn
from sentence_transformers import SentenceTransformer


class FileStats:
	def __init__(self, args):
		self.score_file = args.score_file
		self.src_file = args.src_file
		self.tgt_file = args.tgt_file
		self.step_size = args.step_size
		self.bins = self.prepare_score_dict()
		self.analyze()

	def analyze(self):
		"""
			Retrieve Corpus Statistics, including
			1. distribution of sentence length based on work
			2. distribution of sentence length based on character
			2. count of small segments
			3. count of punctuations of various types, normalized by sentence length
			4. count of sentences with numerical expression
			5. count of sentence that has different numerical expression on both side
			6. count of sentence that has repetition of words
			7. count of sentence that has english tok in other language side
			8. count of sentence that has other language in english side
			9. entropy of english side based on language model
			10. consecutive repetitions of the same character
			11. consecutive repetitions of the same word
		"""
		NUM_SENTENCES, SMALL_SEGMENTS, MEDIUM_SEGMENTS, PUNC_SENTS, NUMBERS, DIFF_NUMBERS, REP_WORDSS, EN_IN_OTHERS, \
		OTHER_IN_ENS, CONS_WORDS, CONS_CHARS, AVG_SCORES, sort_lengths, sort_char_lengths, sort_puncs = \
			[],[],[],[],[],[],[], [], [], [], [], [], [], [], []
		for i, bin in enumerate(self.bins):
			NUM_SENTENCE, SMALL_SEGMENT, MEDIUM_SEGMENT, PUNC_SENT, NUMBER, DIFF_NUMBER, REP_WORDS, EN_IN_OTHER, \
			OTHER_IN_EN, \
			CONS_WORD, CONS_CHAR, AVG_SCORE, sort_length, sort_char_length, sort_punc = self.analyze_bin(bin)
			NUM_SENTENCES.append(NUM_SENTENCE)
			SMALL_SEGMENTS.append(SMALL_SEGMENT)
			MEDIUM_SEGMENTS.append(MEDIUM_SEGMENT)
			PUNC_SENTS.append(PUNC_SENT)
			NUMBERS.append(NUMBER)
			DIFF_NUMBERS.append(DIFF_NUMBER)
			REP_WORDSS.append(REP_WORDS)
			EN_IN_OTHERS.append(EN_IN_OTHER)
			OTHER_IN_ENS.append(OTHER_IN_EN)
			CONS_WORDS.append(CONS_WORD)
			CONS_CHARS.append(CONS_CHAR)
			AVG_SCORES.append(AVG_SCORE)
			sort_lengths.append(sort_length)
			sort_char_lengths.append(sort_char_length)
			sort_puncs.append(sort_punc)

		for i in range(1, len(NUM_SENTENCES)):
			SMALL_SEGMENTS[i] += SMALL_SEGMENTS[i-1]
			MEDIUM_SEGMENTS[i] += MEDIUM_SEGMENTS[i-1]
			PUNC_SENTS[i] += PUNC_SENTS[i-1]
			NUMBERS[i] += NUMBERS[i-1]
			DIFF_NUMBERS[i] += DIFF_NUMBERS[i-1]
			REP_WORDSS[i] += REP_WORDSS[i-1]
			EN_IN_OTHERS[i] += EN_IN_OTHERS[i-1]
			OTHER_IN_ENS[i] += OTHER_IN_ENS[i-1]
			CONS_WORDS[i] += CONS_WORDS[i-1]
			CONS_CHARS[i] += CONS_CHARS[i-1]
		num_sent=0
		for i in range(len(NUM_SENTENCES)):
			num_sent += NUM_SENTENCES[i]
			SMALL_SEGMENTS[i] /= num_sent
			MEDIUM_SEGMENTS[i] /= num_sent
			PUNC_SENTS[i] /= num_sent
			NUMBERS[i] /= num_sent
			DIFF_NUMBERS[i] /= num_sent
			REP_WORDSS[i] /= num_sent
			EN_IN_OTHERS[i] /= num_sent
			OTHER_IN_ENS[i] /= num_sent
			CONS_WORDS[i] /= num_sent
			CONS_CHARS[i] /= num_sent

		print("Number of Sentences: {}\n"
		      "Number of Small Segments (len<3): {}\n"
		      "Number of Small Segments (2<len<5): {}\n"
		      "Number of Irregular Punctuations: {}\n"
		      "Sentences with numbers: {}\n"
		      "Sentence Pairs with Different Numbers: {}\n"
		      "Sentences with Repetitive Words: {}\n"
		      "Sentences with EN in Other: {}\n"
		      "Sentences with Other in EN: {}\n"
		      "Sentences with Consecutive Words:{}\n"
		      "Sentences with Consecutive Chars (>=3) {}\n"
		      "Average Score: {}\n".format(
			" \\ ".join(map(lambda x : str(x), NUM_SENTENCES)),
			" \\ ".join(map(lambda x : format(x, ".3f"), SMALL_SEGMENTS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), MEDIUM_SEGMENTS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), PUNC_SENTS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), NUMBERS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), DIFF_NUMBERS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), REP_WORDSS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), EN_IN_OTHERS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), OTHER_IN_ENS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), CONS_WORDS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), CONS_CHARS)),
			" \\ ".join(map(lambda x : format(x, ".3f"), AVG_SCORES))
		))
		print("------------------------------------------")
		print("TOP 10 Sentence Length and Their Ratio:")
		for k in range(10):
			length_ratio = []
			for i in range(len(sort_lengths)):
				sent_length_dist = sort_lengths[i]
				length, ratio = sent_length_dist[k][0], format(sent_length_dist[k][1]*100/NUM_SENTENCES[i], ".1f")
				length_ratio.append(f"Bin{i+1}: Length {length}, {ratio}%")
			print(" \\ ".join(length_ratio))
		print("------------------------------------------")
		print("TOP 10 Character Length and Their Ratio:")
		for k in range(10):
			length_ratio = []
			for i in range(len(sort_char_lengths)):
				char_length_dist = sort_char_lengths[i]
				length, ratio = char_length_dist[k][0], format(char_length_dist[k][1]*100/NUM_SENTENCES[i], ".1f")
				length_ratio.append(f"Bin{i+1}: Length {length}, {ratio}%")
			print(" \\ ".join(length_ratio))
		print("------------------------------------------")
		print("TOP 10 Punctuations and their frequency:")
		for k in range(10):
			punc_freq = []
			for i in range(len(sort_puncs)):
				punc_dist = sort_puncs[i]
				symbol, freq = punc_dist[k][0], punc_dist[k][1]
				punc_freq.append(f"Bin{i+1}: Symbol {symbol}, {freq}")
			print(" \\ ".join(punc_freq))

	@staticmethod
	def analyze_bin(bin):
		SENTENCE_LENGTH = defaultdict(int)
		CHARACTER_LENGTH = defaultdict(int)
		SMALL_SEGMENT = 0
		MEDIUM_SEGMENT = 0
		PUNCTUATION = defaultdict(int)
		PUNC_SENT = 0
		NUMBER, DIFF_NUMBER = 0, 0
		REP_WORDS = 0
		EN_IN_OTHER, OTHER_IN_EN = 0, 0
		CONS_WORD, CONS_CHAR = 0, 0
		TOTAL_SCORE = 0
		for language_pair, score in bin:
			en, other = language_pair
			TOTAL_SCORE += float(score)
			# length related information
			en_sent_length = len(en.split(' '))
			SENTENCE_LENGTH[en_sent_length] += 1
			en_char_length = len(set(en))
			CHARACTER_LENGTH[en_char_length] += 1
			SMALL_SEGMENT += 1 if en_sent_length < 3 else 0
			MEDIUM_SEGMENT += 1 if (5 > en_sent_length > 2) else 0
			# punctuation related info
			if punctuation_info(en, PUNCTUATION): PUNC_SENT += 1
			# numerical info
			NUMBER, DIFF_NUMBER = number_info(en, other, NUMBER, DIFF_NUMBER)
			# check if words repeat in sentence above a threshold compared to #words
			if rep_words(en): REP_WORDS += 1
			# check if language of opposite side appear in each other
			EN_IN_OTHER, OTHER_IN_EN = wrong_lang(en, other, EN_IN_OTHER, OTHER_IN_EN)
			# TODO: LM for english side entropy
			# consecutive repetition of words and character
			CONS_WORD, CONS_CHAR = cons_rep(en, CONS_WORD, CONS_CHAR)

		sort_length = sorted(SENTENCE_LENGTH.items(), key=lambda x: x[1], reverse=True)[:10]
		sort_char_length = sorted(CHARACTER_LENGTH.items(), key=lambda x: x[1], reverse=True)[:10]
		sort_punc = sorted(PUNCTUATION.items(), key=lambda x: x[1], reverse=True)[:10]

		return len(bin), SMALL_SEGMENT, MEDIUM_SEGMENT, PUNC_SENT, NUMBER, DIFF_NUMBER, REP_WORDS, EN_IN_OTHER, \
		       OTHER_IN_EN, CONS_WORD, CONS_CHAR, TOTAL_SCORE/len(bin), sort_length, sort_char_length, sort_punc

	def prepare_score_dict(self):
		""" sort src tgt sentences by score """
		with open(self.score_file, 'r') as f_score:
			scores = f_score.read()
		with open(self.src_file, 'r') as f_src:
			src_data = f_src.read().split('\n')
		with open(self.tgt_file, 'r') as f_tgt:
			tgt_data = f_tgt.read().split('\n')
		score_dict = {}
		for i, score_line in enumerate(scores.split('\n')):
			if len(score_line) < 1:continue # remove the last empty line
			line = score_line.split('\t')
			if len(line) == 1: # need to concate with src tgt file
				score = line[0]
				score_dict[(tgt_data[i], src_data[i])] = float(score)
			else:
				score, en, other = line[0], line[1], line[2]
				score_dict[(en, other)] = float(score)
		# sort by score
		sorted_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
		# Divide the data into several bins based on step size
		count = 0
		all_bins = []
		current_bin = []
		for (k,v) in sorted_dict:
			current_bin.append((k,v))
			en_sentence = k[0]
			sentence_length = len(en_sentence.split(' '))
			count += sentence_length
			if count >= self.step_size:
				all_bins.append(current_bin)
				current_bin = []
				count = 0
		# ignore the last small bin for the ease of data analysis
		# if len(current_bin) > 0:
		# 	all_bins.append(current_bin)
		return all_bins

def cons_rep(en, cons_word, cons_char):
	clean_text = ""
	for char in en:
		if char not in string.punctuation:
			clean_text += char
	words = clean_text.split(' ')
	for i in range(len(words)-1):
		if words[i] == words[i+1]:
			cons_word += 1
			break
	clean_char = ""
	for tok in en:
		if not ((tok in string.punctuation) or (tok in string.whitespace) or tok.isnumeric()):
			clean_char += tok
	for i in range(len(clean_char)-2):
		# 2 consecutive char is not rare, so only count 3 consecutive chars
		if clean_char[i] == clean_char[i+1] and clean_char[i] == clean_char[i+2]:
			cons_char += 1
			break
	return cons_word, cons_char


def wrong_lang(en, other, en_in_other, other_in_en):
	has_non_english_char = False
	has_eng_char = False
	for en_char in en:
		if not (en_char.isalpha() or en_char.isnumeric() or (en_char in string.punctuation) or (en_char in string.whitespace)):
			has_non_english_char = True
	other_in_en += 1 if has_non_english_char else 0
	for other_char in other:
		if other_char in "abcdefghijklmnopqrstuvwxyz":
			has_eng_char = True
	en_in_other += 1 if has_eng_char else 0
	return en_in_other, other_in_en


def rep_words(en):
	clean_text = ""
	for char in en:
		if char not in string.punctuation:
			clean_text += char
	words = clean_text.split(' ')
	return  (len(words) - len(set(words))) / len(words) > 0.25

def punctuation_info(en, PUNCTUATION):
	has_punc = False
	for char in en:
		# check punctuation
		if char in string.punctuation:
			PUNCTUATION[char] += 1
			if not (char == ',' or char == '.' or char == '?' or char == '!'):
				# non regular punctuations
				has_punc = True
	return has_punc

def number_info(en, other, NUMBER, DIFF_NUMBER):
	has_number = False
	number_appeared = []
	for char in en:
		if char.isnumeric():
			has_number = True
			number_appeared.append(char)
	if has_number:
		NUMBER += 1
		# check if other side has the same numbers, count as yes if half of numbers appear in other side
		overlap_number = 0
		for char in other:
			if char.isnumeric():
				if char in number_appeared:
					overlap_number += 1
		if overlap_number < 0.5 * len(number_appeared):
			DIFF_NUMBER += 1
	return NUMBER, DIFF_NUMBER

def parser_args():
	parser = argparse.ArgumentParser(description='Use Laser score file to output the bilingual dataset')
	parser.add_argument('--score-file', required=True, default='.', type=str, help='score file')
	parser.add_argument('--src-file', required=True, type=str)
	parser.add_argument('--tgt-file', required=True, type=str)
	parser.add_argument('--step-size', default=1e6, type=float, help="analyze the corpus based on the step/bin size")
	# parser.add_argument('--output', required=True, default='.', type=str, help='output bilingual file for NMT')
	# parser.add_argument('--lang', required=True, type=str)
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parser_args()
	FileStats(args)
