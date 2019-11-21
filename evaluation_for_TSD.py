# coding=utf-8

"""evaluate P R F1 for target & polarity joint task"""

import csv
import os
import re
import pandas as pd

def TXT_file(name):
	return '{}.txt'.format(name)

def Clean_file(name):
	return '{}.tsv'.format(name)

path = 'results/semeval2015/three_joint/BIO/prefix_18batch'
epochs = 30	# epoch num

for index in range(24, 25):
	file_pre = 'test_ep_' + str(index)
	with open(os.path.join(path, TXT_file(file_pre)), 'r', encoding='utf-8') as f_pre:
		Gold_Num = 0
		True_Num = 0
		Pre_Num = 0
		if '/TO/' in path:
			entity_label = r"T+" # for TO
		else:
			entity_label = r"BI*" # for BIO
		f_pre.readline()
		pre_lines = f_pre.readlines()

		# the polarity order in test file is: positive, negative, neutral
		lin_idx = 0
		positive_targets_gold = set()
		positive_targets_pred = set()
		negative_targets_gold = set()
		negative_targets_pred = set()
		neutral_targets_gold = set()
		neutral_targets_pred = set()
		NULL_for_positive_gold = False
		NULL_for_positive_pred = False
		NULL_for_negative_gold = False
		NULL_for_negative_pred = False
		NULL_for_neutral_gold = False
		NULL_for_neutral_pred = False
		pre_sen = ''
		now_sen = ''
		for line in pre_lines:
			lin_idx += 1
			pre_line = line.strip().split('\t')
			now_sen = pre_line[2]
			if now_sen != pre_sen:	# a new sentence now, evaluate for pre sentence
				pre_sen = now_sen
				# positive
				'''
				if NULL_for_positive_gold:
					Gold_Num += 1
				if NULL_for_positive_pred:
					Pre_Num += 1
				if NULL_for_positive_gold and NULL_for_positive_pred:
					True_Num += 1
				'''
				Gold_Num += len(positive_targets_gold)
				Pre_Num += len(positive_targets_pred)
				True_Num += len(positive_targets_gold & positive_targets_pred)
				# negative
				'''
				if NULL_for_negative_gold:
					Gold_Num += 1
				if NULL_for_negative_pred:
					Pre_Num += 1
				if NULL_for_negative_gold and NULL_for_negative_pred:
					True_Num += 1
				'''
				Gold_Num += len(negative_targets_gold)
				Pre_Num += len(negative_targets_pred)
				True_Num += len(negative_targets_gold & negative_targets_pred)
				# neutral
				'''
				if NULL_for_neutral_gold:
					Gold_Num += 1
				if NULL_for_neutral_pred:
					Pre_Num += 1
				if NULL_for_neutral_gold and NULL_for_neutral_pred:
					True_Num += 1
				'''
				Gold_Num += len(neutral_targets_gold)
				Pre_Num += len(neutral_targets_pred)
				True_Num += len(neutral_targets_gold & neutral_targets_pred)

				# initialize for new sentence
				positive_targets_gold.clear()
				positive_targets_pred.clear()
				negative_targets_gold.clear()
				negative_targets_pred.clear()
				neutral_targets_gold.clear()
				neutral_targets_pred.clear()
				NULL_for_positive_gold = False
				NULL_for_positive_pred = False
				NULL_for_negative_gold = False
				NULL_for_negative_pred = False
				NULL_for_neutral_gold = False
				NULL_for_neutral_pred = False

			sentence_length = len(pre_line[2].split())
			pre_ner_tags = ''.join(pre_line[-1].split()[1:])	# [CLS] sentence [SEP] ........
			gold_ner_tags = ''.join(pre_line[-2].split()[1:])
			if pre_line[0] == '1':	# yes on gold
				gold_entity = set()
				gold_entity_list = re.finditer(entity_label, gold_ner_tags)
				for x in gold_entity_list:
					gold_entity.add(str(x.start()) + '-' + str(len(x.group())))

				if lin_idx % 3 == 1:		# this line for positive
					if len(gold_entity) == 0:	# NULL
						NULL_for_positive_gold = True
					else:	# not NULL, has entity in this sentence
						positive_targets_gold = positive_targets_gold | gold_entity
				elif lin_idx % 3 == 2: 	# this line for negative
					if len(gold_entity) == 0:	# NULL
						NULL_for_negative_gold = True
					else:	# not NULL, has entity in this sentence
						negative_targets_gold = negative_targets_gold | gold_entity
				else: 					# this line for neutral
					if len(gold_entity) == 0:	# NULL
						NULL_for_neutral_gold = True
					else:	# not NULL, has entity in this sentence
						neutral_targets_gold = neutral_targets_gold | gold_entity

			if pre_line[1] == '1': # yes on pre
				pre_entity = set()
				pre_entity_list = re.finditer(entity_label, pre_ner_tags)
				for x in pre_entity_list:
					pre_entity.add(str(x.start()) + '-' + str(len(x.group())))

				if lin_idx % 3 == 1:		# this line for positive
					if len(pre_entity) == 0:	# NULL
						NULL_for_positive_pred = True
					else:	# not NULL, has entity in this sentence
						positive_targets_pred = positive_targets_pred | pre_entity
				elif lin_idx % 3 == 2: 	# this line for negative
					if len(pre_entity) == 0:	# NULL
						NULL_for_negative_pred = True
					else:	# not NULL, has entity in this sentence
						negative_targets_pred = negative_targets_pred | pre_entity
				else: 					# this line for neutral
					if len(pre_entity) == 0:	# NULL
						NULL_for_neutral_pred = True
					else:	# not NULL, has entity in this sentence
						neutral_targets_pred = neutral_targets_pred | pre_entity

		P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
		R = True_Num / float(Gold_Num)
		F = (2*P*R)/float(P+R) if P!=0 else 0
		print("\tP: ", P, "   R: ", R, "  F1: ", F)
		print('\t\tgold sum: ', Gold_Num)
		print('\t\tpre  sum: ', Pre_Num)
		print('\t\ttrue sum: ', True_Num)
		print('----------------------------------------------------\n')
		



