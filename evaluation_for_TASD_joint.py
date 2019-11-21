# coding=utf-8

"""evaluate P R F1 for:
		1. three joint task
		2. target & polarity joint task
		3. target & category joint task
"""

import csv
import os
import re
import pandas as pd

def TXT_file(name):
	return '{}.txt'.format(name)

def Clean_file(name):
	return '{}.tsv'.format(name)

path = 'results/semeval2016/three_joint/BIO/prefix_18batch'
epochs = 30 # epoch num
best_F = 0
best_ep = 0

for index in range(1, epochs+1):
	file_pre = 'test_ep_' + str(index)
	with open(os.path.join(path, TXT_file(file_pre)), 'r', encoding='utf-8') as f_pre:
		Gold_Num = 0
		True_Num = 0
		Pre_Num = 0
		NULL_Gold_Num = 0
		NULL_True_Num = 0
		NULL_Pre_Num = 0
		NO_and_O_Gold_Num = 0
		NO_and_O_True_Num = 0
		NO_and_O_Pre_Num = 0
		if '/TO/' in path:
			entity_label = r"T+" # for TO
		else:
			entity_label = r"BI*" # for BIO
		f_pre.readline()
		pre_lines = f_pre.readlines()
		for line in pre_lines:
			pre_line = line.strip().split('\t')
			sentence_length = len(pre_line[2].split())
			pre_ner_tags = ''.join(pre_line[-1].split()[1:])	# [CLS] sentence [SEP] ........
			gold_ner_tags = ''.join(pre_line[-2].split()[1:])
			if pre_line[0] == '1':	# yes on gold
				gold_entity = []
				pre_entity = []
				gold_entity_list = re.finditer(entity_label, gold_ner_tags)
				pre_entity_list = re.finditer(entity_label, pre_ner_tags)
				for x in gold_entity_list:
					gold_entity.append(str(x.start()) + '-' + str(len(x.group())))
				for x in pre_entity_list:
					pre_entity.append(str(x.start()) + '-' + str(len(x.group())))

				if len(gold_entity) == 0:	# NULL
					Gold_Num += 1
					NULL_Gold_Num += 1
					if len(pre_entity) == 0 and pre_line[1] == '1':
						True_Num += 1
						NULL_True_Num += 1
				else:	# not NULL, has entity in this sentence
					Gold_Num += len(gold_entity)
					for x in gold_entity:
						if x in pre_entity and pre_line[1] == '1':
							True_Num += 1
			else:	# no on gold
				NO_and_O_Gold_Num += 1
				if pre_line[1] == '0' and 'T' not in pre_ner_tags and 'B' not in pre_ner_tags and 'I' not in pre_ner_tags:
					NO_and_O_True_Num += 1

			if pre_line[1] == '1': # yes on pre
				pre_entity = []
				pre_entity_list = re.finditer(entity_label, pre_ner_tags)
				for x in pre_entity_list:
					pre_entity.append(str(x.start()) + '-' + str(len(x.group())))

				if len(pre_entity) == 0:	# NULL
					Pre_Num += 1
					NULL_Pre_Num += 1
				else:	# not NULL, has entity in this sentence
					Pre_Num += len(pre_entity)
			else:	# no on pre
				if 'T' not in pre_ner_tags and 'B' not in pre_ner_tags and 'I' not in pre_ner_tags:
					NO_and_O_Pre_Num += 1

		P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
		R = True_Num / float(Gold_Num)
		F = (2*P*R)/float(P+R) if P!=0 else 0

		P_NULL = NULL_True_Num / float(NULL_Pre_Num) if NULL_Pre_Num != 0 else 0
		R_NULL = NULL_True_Num / float(NULL_Gold_Num)
		F_NULL = (2*P_NULL*R_NULL)/float(P_NULL+R_NULL) if P_NULL!=0 else 0

		P_NO_and_O = NO_and_O_True_Num / float(NO_and_O_Pre_Num) if NO_and_O_Pre_Num != 0 else 0
		R_NO_and_O = NO_and_O_True_Num / float(NO_and_O_Gold_Num)
		F_NO_and_O = (2*P_NO_and_O*R_NO_and_O)/float(P_NO_and_O+R_NO_and_O) if P_NO_and_O!=0 else 0

		if F > best_F:
			best_F = F
			best_ep = index

		print('ep_', index)
		print('All tuples')
		print("\tP: ", P, "   R: ", R, "  F1: ", F)
		print('\t\tgold sum: ', Gold_Num)
		print('\t\tpre  sum: ', Pre_Num)
		print('\t\ttrue sum: ', True_Num)
		print('----------------------------------------------------\n')

		print('Only NULL tuples')
		print("\tP: ", P_NULL, "   R: ", R_NULL, "  F1: ", F_NULL)
		print('\t\tgold sum: ', NULL_Gold_Num)
		print('\t\tpre  sum: ', NULL_Pre_Num)
		print('\t\ttrue sum: ', NULL_True_Num)
		print('----------------------------------------------------\n')

		print('NO and pure O tag sequence')
		print("\tP: ", P_NO_and_O, "   R: ", R_NO_and_O, "  F1: ", F_NO_and_O)
		print('\t\tgold sum: ', NO_and_O_Gold_Num)
		print('\t\tpre  sum: ', NO_and_O_Pre_Num)
		print('\t\ttrue sum: ', NO_and_O_True_Num)
		print('----------------------------------------------------\n')

print('best: ep_', best_ep)

