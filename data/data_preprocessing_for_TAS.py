# coding=utf-8

"""
Preprocessing for SemEval-2015 and SemEval-2016 Datasets
			  for three joint task (target & aspect & sentiment)

TO floder means TO labeling schema for targets
BIO floder means BIO labeling schema for targets
"""

import csv
import os
import re
import argparse
from change_TO_to_BIO import TXT_file, TSV_file, change_TO_to_BIO

def get_aspect_sentiment_compose(path, file_name):
	aspect_set = []
	sentiment_set = ['positive', 'negative', 'neutral'] # sentiment polarity
	with open(os.path.join(path, TXT_file(file_name)), 'r', encoding='utf-8') as fin:
		fin.readline()
		for line in fin:
			line_arr = line.strip().split('\t')
			if line_arr[6] == 'yes':	# entailed == yes
				if line_arr[3] not in aspect_set:
					aspect_set.append(line_arr[3])	# aspect

	compose_set = []
	for ca in aspect_set:
		for po in sentiment_set:
			compose_set.append(ca + ' ' + po)

	return compose_set


def create_dataset_file(input_path, output_path, input_file, output_file, compose_set):
	one_data_nums = 0
	zero_data_nums = 0
	max_len = 0
	entity_sum = 0
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	with open(os.path.join(input_path, TXT_file(input_file)), 'r', encoding='utf-8') as fin, open(os.path.join(output_path, TSV_file(output_file)), 'w', encoding='utf-8') as fout:
		fout.write('\t'.join(['sentence_id', 'yes_no', 'aspect_sentiment', 'sentence', 'ner_tags']))
		fout.write('\n')
		fin.readline()
		pre_start = False
		pre_sentence_id = 'XXX'	# sentence id of the previous line
		pre_sentence = 'XXX'
		record_of_one_sentence = set()	# the set of aspect&sentiment that this sentence contains
		record_of_one_sentence_ner_tag = {}	# the NER tags of the set of aspect&sentiment that this sentence contains
		for line in fin:
			line_arr = line.strip().split('\t')
			sentence_id = line_arr[0]
			if sentence_id != pre_sentence_id:	# this is a new sentence
				if pre_start == True:
					for x in compose_set:
						# create yes line of this sentence
						if x in record_of_one_sentence:
							fout.write(pre_sentence_id + '\t' + '1' + '\t' + x + '\t' + pre_sentence + '\t' + record_of_one_sentence_ner_tag[x] + '\n')
							one_data_nums += 1
						# create no line
						else:
							fout.write(pre_sentence_id + '\t' + '0' + '\t' + x + '\t' + pre_sentence + '\t' + ' '.join(['O']*len(pre_sentence.split())) + '\n')
							zero_data_nums += 1

				else:
					pre_start = True
				record_of_one_sentence.clear()
				record_of_one_sentence_ner_tag.clear()
				pre_sentence_id = sentence_id


			if line_arr[6] == 'yes':	# entailed == yes
				# get NER labels
				sentence = line_arr[1].strip().split(' ')
				gold_target = ' '.join(line_arr[2].strip().split())
				ner_tags = ['O'] * len(sentence)
				start = int(line_arr[7]) - 1
				end = int(line_arr[8]) - 1
				if line_arr[1].startswith(' '):
					start -= 1
					end -= 1
				if not (start < 0 and end < 0):	# not NULL
					get_target = ' '.join(sentence[start:end])
					if gold_target != get_target:
						print('Error!!!!')
						print(line_arr[1])
						print(gold_target)
						print(get_target)
						print(str(start) + ' - ' + str(end))

					for x in range(start, end):
						ner_tags[x] = 'T'

				sentence_clear = []
				ner_tags_clear = []
				# solve the '  ' multi space
				special_token = "$()*+.[]?\\^}{|!'#%&,-/:;_~@<=>`\"’“”‘…"
				special_token_re = r"[\$\(\)\*\+\.\[\]\?\\\^\{\}\|!'#%&,-/:;_~@<=>`\"’‘“”…]{1,1}"
				for x in range(len(sentence)):
					in_word = False
					if sentence[x] != '':
						punctuation_list = re.finditer(special_token_re, sentence[x])
						punctuation_list_start = []
						punctuation_list_len = []
						for m in punctuation_list:
							punctuation_list_start.append(m.start())
							punctuation_list_len.append(len(m.group()))

						if len(punctuation_list_start) != 0:
							# the start is word
							if punctuation_list_start[0] != 0:
								sentence_clear.append(sentence[x][0:punctuation_list_start[0]])
								ner_tags_clear.append(ner_tags[x])
							for (i, m) in enumerate(punctuation_list_start):
								#print(len(punctuation_list_start))
								#print(len(punctuation_list_len))
								#print(str(m) + ' - ' + str(m+punctuation_list_len[i]))
								sentence_clear.append(sentence[x][m:m+punctuation_list_len[i]])
								ner_tags_clear.append(ner_tags[x])

								if i != len(punctuation_list_start) - 1:
									if m+punctuation_list_len[i] != punctuation_list_start[i+1] :
										sentence_clear.append(sentence[x][m+punctuation_list_len[i]:punctuation_list_start[i+1]])
										ner_tags_clear.append(ner_tags[x])

								else:
									if m+punctuation_list_len[i] < len(sentence[x]):
										sentence_clear.append(sentence[x][m+punctuation_list_len[i]:])
										ner_tags_clear.append(ner_tags[x])


						else: # has no punctuation
							sentence_clear.append(sentence[x])
							ner_tags_clear.append(ner_tags[x])

				assert '' not in sentence_clear
				assert len(sentence_clear) == len(ner_tags_clear)

				# get aspect&sentiment
				cate_pola = line_arr[3] + ' ' + line_arr[4]

				pre_sentence = ' '.join(sentence_clear)
				assert '  ' not in pre_sentence

				if len(sentence_clear) > max_len:
					max_len = len(sentence_clear)
				if cate_pola in record_of_one_sentence:	# this aspect&sentiment has more than one target
					ner_tags_A = record_of_one_sentence_ner_tag[cate_pola].split()
					if len(ner_tags_A) != len(ner_tags_clear):
						print('Ner Tags Length Error!!!')
					else:
						for x in range(len(ner_tags_A)):
							if ner_tags_A[x] != 'O':
								ner_tags_clear[x] = ner_tags_A[x]
						record_of_one_sentence_ner_tag[cate_pola] = ' '.join(ner_tags_clear)
						if 'T' in ner_tags_A:
							entity_sum += 1

				else:
					record_of_one_sentence.add(cate_pola)
					record_of_one_sentence_ner_tag[cate_pola] = ' '.join(ner_tags_clear)
					entity_sum += 1
	print('entity_sum: ', entity_sum)
	print('max_sen_len: ', max_len)
	print('sample ratio: ', str(one_data_nums), '-', str(zero_data_nums))




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset',
						type=str,
						choices=["semeval2015", "semeval2016"],
						help='dataset, as a folder name, you can choose from semeval2015 and semeval2016')
	args = parser.parse_args()

	path = args.dataset + '/three_joint'
	output_path = path + '/TO'

	if '2015' in args.dataset:
		train_file = 'ABSA_15_Restaurants_Train'
		test_file = 'ABSA_15_Restaurants_Test'
	else:
		train_file = 'ABSA_16_Restaurants_Train'
		test_file = 'ABSA_16_Restaurants_Test'

	train_output = 'train_TAS'
	test_output = 'test_TAS'

	# get set of aspect-sentiment
	compose_set = get_aspect_sentiment_compose(args.dataset, train_file)

	for input_file, output_file in zip([train_file, test_file], [train_output, test_output]):
		# get preprocessed data, TO labeling schema
		create_dataset_file(args.dataset, output_path, input_file, output_file, compose_set)
		# get preprocessed data, BIO labeling schema
		change_TO_to_BIO(path, output_file)

