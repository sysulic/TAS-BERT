# coding=utf-8

"""
create BIO labels for targets
"""

import csv
import os
import re

def TXT_file(name):
	return '{}.txt'.format(name)

def TSV_file(name):
	return '{}.tsv'.format(name)


def change_TO_to_BIO(path, file_name):
	input_path = path + '/TO'
	entity_label = r"T+"
	output_path = path + '/BIO'
	file_out = file_name
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	with open(os.path.join(input_path, TSV_file(file_name)), 'r', encoding='utf-8') as fin, open(os.path.join(output_path, TSV_file(file_out)), 'w', encoding='utf-8') as fout:
		fout.write('\t'.join(['sentence_id', 'yes_no', 'aspect_sentiment', 'sentence', 'ner_tags']))
		fout.write('\n')
		fin.readline()
		for line in fin:
			line_arr = line.strip().split('\t')
			# change TO to BIO tags
			ner_tags = ''.join(line_arr[-1].split())
			entity_list = re.finditer(entity_label, ner_tags)
			BIO_tags = ['O'] * len(ner_tags)
			for x in entity_list:
				start = x.start()
				en_len = len(x.group())
				BIO_tags[start] = 'B'
				for m in range(start+1, start+en_len):
					BIO_tags[m] = 'I'

			line_arr[-1] = ' '.join(BIO_tags)
			fout.write('\t'.join(line_arr))
			fout.write('\n')