# coding=utf-8

"""preprocess for dataset."""

import csv
import os
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOM

def TXT_file(name):
	return '{}.txt'.format(name)

def TSV_file(name):
	return '{}.tsv'.format(name)

def XML_file(name):
	return '{}.xml'.format(name)

gold_path = '../data/semeval2016/three_joint/BIO'
file_gold = 'test_NLI_B'
pre_path = '../results/semeval2016/category_polarity/word_2'
file_pre = 'test_ep_23'
file_xml = 'ABSA16_Restaurants_Test.xml' # gold xml file -> used to get the file structure

if 'BIO' in gold_path:
	entity_label = r"BI*" # for BIO
else:
	entity_label = r"T+" # for TO



with open(os.path.join(pre_path, TXT_file(file_pre)), 'r', encoding='utf-8') as f_pre, open(os.path.join(gold_path, TSV_file(file_gold)), 'r', encoding='utf-8') as f_gold:
	# clear the gold opinions and get the empty framework
	sen_tree_map = {}
	xml_tree = ET.parse(file_xml)
	root = xml_tree.getroot()
	for review in root.findall('Review'):
		for sen in review.find('sentences').getchildren():
			sen_key = sen.get('id')
			sen_tree_map[sen_key] = sen
			opinions = sen.find('Opinions')
			if opinions is not None:
				opinions.clear()
	
	f_pre.readline()
	f_gold.readline()
	pre_lines = f_pre.readlines()
	gold_lines = f_gold.readlines()
	for line_1, line_2 in zip(pre_lines, gold_lines):
		pre_line = line_1.strip().split('\t')
		gold_line = line_2.strip().split('\t')
		
		sentence_id = gold_line[0]
		yes_not = pre_line[1]
		category_polarity = gold_line[2].split()
		sentence = gold_line[3].split()

		if yes_not == '1':
			category = '#'.join(category_polarity[0:2]).upper()
			polarity = category_polarity[-1]
			

			# write to the xml file
			
			current_sen = sen_tree_map[sentence_id]
			current_opinions = current_sen.find('Opinions')
			if current_opinions == None:
				current_opinions = ET.Element('Opinions')
				current_sen.append(current_opinions)

			op = ET.Element('Opinion')
			op.set('target', 'NULL')
			op.set('category', category)
			op.set('polarity', polarity)
			op.set('from', '0')
			op.set('to', '0')
			current_opinions.append(op)

	xml_string = ET.tostring(root)
	xml_write = DOM.parseString(xml_string)
	with open('pred.xml', 'w') as handle:
		xml_write.writexml(handle, indent=' ', newl='\n', encoding='utf-8')
