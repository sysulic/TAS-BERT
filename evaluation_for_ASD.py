# coding=utf-8

"""evaluate P R F1 for category & polarity joint task"""

import csv
import os
import re
import pandas as pd

def TXT_file(name):
	return '{}.txt'.format(name)

def Clean_file(name):
	return '{}.tsv'.format(name)

path = 'results/semeval2016/category_polarity/word_2'
epochs = 30	# epoch num

for index in range(1, epochs+1):
	file_pre = 'test_ep_' + str(index)
	with open(os.path.join(path, TXT_file(file_pre)), 'r', encoding='utf-8') as f_pre:
		Gold_Num = 0
		True_Num = 0
		Pre_Num = 0
		f_pre.readline()
		pre_lines = f_pre.readlines()
		for line in pre_lines:
			pre_line = line.strip().split('\t')

			if pre_line[0] == '1':	# yes on gold
				Gold_Num += 1
				if pre_line[1] == '1': # yes on pre
					True_Num += 1

			if pre_line[1] == '1': # yes on pre
				Pre_Num += 1

		P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
		R = True_Num / float(Gold_Num)
		F = (2*P*R)/float(P+R) if P!=0 else 0
		print('ep_', index)
		print("\tP: ", P, "   R: ", R, "  F1: ", F)
		print('\t\tgold sum: ', Gold_Num)
		print('\t\tpre  sum: ', Pre_Num)
		print('\t\ttrue sum: ', True_Num)
		print('----------------------------------------------------\n')


