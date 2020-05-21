# coding=utf-8

"""evaluate P R F1 for target & polarity joint task"""

import csv
import os
import re
import argparse

def TXT_file(name):
	return '{}.txt'.format(name)

def Clean_file(name):
	return '{}.tsv'.format(name)

def evaluate_TSD_contain_NULL(path, best_epoch_file, tag_schema):
	with open(os.path.join(path, TXT_file(best_epoch_file)), 'r', encoding='utf-8') as f_pre:
		Gold_Num = 0
		True_Num = 0
		Pre_Num = 0
		tag_schema == 'TO'
		if tag_schema == 'TO':
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
			if now_sen != pre_sen:  # a new sentence now, evaluate for pre sentence
				pre_sen = now_sen
				# positive
				if NULL_for_positive_gold:
					Gold_Num += 1
				if NULL_for_positive_pred:
					Pre_Num += 1
				if NULL_for_positive_gold and NULL_for_positive_pred:
					True_Num += 1
				Gold_Num += len(positive_targets_gold)
				Pre_Num += len(positive_targets_pred)
				True_Num += len(positive_targets_gold & positive_targets_pred)
				# negative
				if NULL_for_negative_gold:
					Gold_Num += 1
				if NULL_for_negative_pred:
					Pre_Num += 1
				if NULL_for_negative_gold and NULL_for_negative_pred:
					True_Num += 1
				Gold_Num += len(negative_targets_gold)
				Pre_Num += len(negative_targets_pred)
				True_Num += len(negative_targets_gold & negative_targets_pred)
				# neutral
				if NULL_for_neutral_gold:
					Gold_Num += 1
				if NULL_for_neutral_pred:
					Pre_Num += 1
				if NULL_for_neutral_gold and NULL_for_neutral_pred:
					True_Num += 1
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
			pre_ner_tags = ''.join(pre_line[-1].split()[1:])    # [CLS] sentence [SEP] ........
			gold_ner_tags = ''.join(pre_line[-2].split()[1:])
			if pre_line[0] == '1':  # yes on gold
				gold_entity = set()
				gold_entity_list = re.finditer(entity_label, gold_ner_tags)
				for x in gold_entity_list:
					gold_entity.add(str(x.start()) + '-' + str(len(x.group())))

				if lin_idx % 3 == 1:        # this line for positive
					if len(gold_entity) == 0:   # NULL
						NULL_for_positive_gold = True
					else:   # not NULL, has entity in this sentence
						positive_targets_gold = positive_targets_gold | gold_entity
				elif lin_idx % 3 == 2:  # this line for negative
					if len(gold_entity) == 0:   # NULL
						NULL_for_negative_gold = True
					else:   # not NULL, has entity in this sentence
						negative_targets_gold = negative_targets_gold | gold_entity
				else:                   # this line for neutral
					if len(gold_entity) == 0:   # NULL
						NULL_for_neutral_gold = True
					else:   # not NULL, has entity in this sentence
						neutral_targets_gold = neutral_targets_gold | gold_entity

			if pre_line[1] == '1': # yes on pre
				pre_entity = set()
				pre_entity_list = re.finditer(entity_label, pre_ner_tags)
				for x in pre_entity_list:
					pre_entity.add(str(x.start()) + '-' + str(len(x.group())))

				if lin_idx % 3 == 1:        # this line for positive
					if len(pre_entity) == 0:    # NULL
						NULL_for_positive_pred = True
					else:   # not NULL, has entity in this sentence
						positive_targets_pred = positive_targets_pred | pre_entity
				elif lin_idx % 3 == 2:  # this line for negative
					if len(pre_entity) == 0:    # NULL
						NULL_for_negative_pred = True
					else:   # not NULL, has entity in this sentence
						negative_targets_pred = negative_targets_pred | pre_entity
				else:                   # this line for neutral
					if len(pre_entity) == 0:    # NULL
						NULL_for_neutral_pred = True
					else:   # not NULL, has entity in this sentence
						neutral_targets_pred = neutral_targets_pred | pre_entity

		P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
		R = True_Num / float(Gold_Num)
		F = (2*P*R)/float(P+R) if P!=0 else 0

		print('TSD task containing NULL:')
		print("\tP: ", P, "   R: ", R, "  F1: ", F)
		print('----------------------------------------------------\n\n')

def evaluate_TSD_ignore_NULL(path, best_epoch_file, tag_schema):
	with open(os.path.join(path, TXT_file(best_epoch_file)), 'r', encoding='utf-8') as f_pre:
		Gold_Num = 0
		True_Num = 0
		Pre_Num = 0
		if tag_schema == 'TO':
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
		pre_sen = ''
		now_sen = ''
		for line in pre_lines:
			lin_idx += 1
			pre_line = line.strip().split('\t')
			now_sen = pre_line[2]
			if now_sen != pre_sen:  # a new sentence now, evaluate for pre sentence
				pre_sen = now_sen
				# positive
				Gold_Num += len(positive_targets_gold)
				Pre_Num += len(positive_targets_pred)
				True_Num += len(positive_targets_gold & positive_targets_pred)
				# negative
				Gold_Num += len(negative_targets_gold)
				Pre_Num += len(negative_targets_pred)
				True_Num += len(negative_targets_gold & negative_targets_pred)
				# neutral
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
			pre_ner_tags = ''.join(pre_line[-1].split()[1:])    # [CLS] sentence [SEP] ........
			gold_ner_tags = ''.join(pre_line[-2].split()[1:])
			if pre_line[0] == '1':  # yes on gold
				gold_entity = set()
				gold_entity_list = re.finditer(entity_label, gold_ner_tags)
				for x in gold_entity_list:
					gold_entity.add(str(x.start()) + '-' + str(len(x.group())))

				if lin_idx % 3 == 1:        # this line for positive
					if len(gold_entity) != 0:   # not NULL, has entity in this sentence
						positive_targets_gold = positive_targets_gold | gold_entity
				elif lin_idx % 3 == 2:  # this line for negative
					if len(gold_entity) != 0:   # not NULL, has entity in this sentence
						negative_targets_gold = negative_targets_gold | gold_entity
				else:                   # this line for neutral
					if len(gold_entity) != 0:   # not NULL, has entity in this sentence
						neutral_targets_gold = neutral_targets_gold | gold_entity

			if pre_line[1] == '1': # yes on pre
				pre_entity = set()
				pre_entity_list = re.finditer(entity_label, pre_ner_tags)
				for x in pre_entity_list:
					pre_entity.add(str(x.start()) + '-' + str(len(x.group())))

				if lin_idx % 3 == 1:        # this line for positive
					if len(pre_entity) != 0:    # not NULL, has entity in this sentence
						positive_targets_pred = positive_targets_pred | pre_entity
				elif lin_idx % 3 == 2:  # this line for negative
					if len(pre_entity) != 0:    # not NULL, has entity in this sentence
						negative_targets_pred = negative_targets_pred | pre_entity
				else:                   # this line for neutral
					if len(pre_entity) != 0:    # not NULL, has entity in this sentence
						neutral_targets_pred = neutral_targets_pred | pre_entity

		P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
		R = True_Num / float(Gold_Num)
		F = (2*P*R)/float(P+R) if P!=0 else 0

		print('TSD task ignoring NULL:')
		print("\tP: ", P, "   R: ", R, "  F1: ", F)
		print('----------------------------------------------------\n\n')


def evaluate_ASD(path, best_epoch_file):
	with open(os.path.join(path, TXT_file(best_epoch_file)), 'r', encoding='utf-8') as f_pre:
		Gold_Num = 0
		True_Num = 0
		Pre_Num = 0
		f_pre.readline()
		pre_lines = f_pre.readlines()
		for line in pre_lines:
			pre_line = line.strip().split('\t')

			if pre_line[0] == '1':  # yes on gold
				Gold_Num += 1
				if pre_line[1] == '1': # yes on pre
					True_Num += 1

			if pre_line[1] == '1': # yes on pre
				Pre_Num += 1

		P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
		R = True_Num / float(Gold_Num)
		F = (2*P*R)/float(P+R) if P!=0 else 0

		print('ASD task:')
		print("\tP: ", P, "   R: ", R, "  F1: ", F)
		print('----------------------------------------------------\n\n')


def evaluate_TASD(path, epochs, tag_schema):
	# record the best epoch
	best_epoch_file = ''
	best_P = 0
	best_R = 0
	best_F1 = 0
	best_NULL_P = 0
	best_NULL_R = 0
	best_NULL_F1 = 0
	best_NO_and_O_P = 0
	best_NO_and_O_R = 0
	best_NO_and_O_F1 = 0
	for index in range(epochs):
		file_pre = 'test_ep_' + str(index+1)
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
			if tag_schema == 'TO':
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

			if F > best_F1:
				best_P = P
				best_R = R
				best_F1 = F

				best_NULL_P = P_NULL
				best_NULL_R = R_NULL
				best_NULL_F1 = F_NULL

				best_NO_and_O_P = P_NO_and_O
				best_NO_and_O_R = R_NO_and_O
				best_NO_and_O_F1 = F_NO_and_O

				best_epoch_file = file_pre

			'''
			print(file_pre, ' :')
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
			'''

	print('\n')
	print("The best result is in ", best_epoch_file, ' :')

	print("TASD task:")
	print("\tAll tuples")
	print("\t\tP: ", best_P, "   R: ", best_R, "  F1: ", best_F1)
	print('----------------------------------------------------\n')
	print("\tOnly NULL tuples")
	print("\t\tP: ", best_NULL_P, "   R: ", best_NULL_R, "  F1: ", best_NULL_F1)
	print('----------------------------------------------------\n')
	print("\tNO and pure O tag sequence")
	print("\t\tP: ", best_NO_and_O_P, "   R: ", best_NO_and_O_R, "  F1: ", best_NO_and_O_F1)
	print('----------------------------------------------------\n\n')

	return best_epoch_file


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--output_dir",
						type=str,
						required=True,
						help="The output_dir in training & testing")
	parser.add_argument("--tag_schema",
						type=str,
						required=True,
						choices=["TO", "BIO"],
						help="The tag schema of the result")
	parser.add_argument("--num_epochs",
						type=int,
						required=True,
						default=30,
						help="The epochs num in training & testing")

	args = parser.parse_args()

	best_epoch_file = evaluate_TASD(args.output_dir, args.num_epochs, args.tag_schema)
	evaluate_ASD(args.output_dir, best_epoch_file)
	evaluate_TSD_contain_NULL(args.output_dir, best_epoch_file, args.tag_schema)
	evaluate_TSD_ignore_NULL(args.output_dir, best_epoch_file, args.tag_schema)



