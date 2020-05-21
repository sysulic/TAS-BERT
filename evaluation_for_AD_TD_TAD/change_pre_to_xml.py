# coding=utf-8

"""preprocess for dataset."""

import csv
import os
import re
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOM

def TXT_file(name):
	return '{}.txt'.format(name)

def TSV_file(name):
	return '{}.tsv'.format(name)

def XML_file(name):
	return '{}.xml'.format(name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--gold_path",
						type=str,
						required=True,
						help="The gold file")
	parser.add_argument("--pre_path",
						type=str,
						required=True,
						help="The test result file, such as */test_ep_21.txt")
	parser.add_argument("--gold_xml_file",
						type=str,
						required=True,
						help="gold xml file -> used to get the file structure")

	parser.add_argument("--pre_xml_file",
						type=str,
						required=True,
						help="get the prediction file in XML format")
	parser.add_argument("--tag_schema",
						type=str,
						required=True,
						choices=["TO", "BIO"],
						help="The tag schema of the result in pre_path")

	args = parser.parse_args()

	if args.tag_schema == 'BIO':
		entity_label = r"BI*" # for BIO
	else:
		entity_label = r"T+" # for TO

	with open(args.pre_path, 'r', encoding='utf-8') as f_pre, open(args.gold_path, 'r', encoding='utf-8') as f_gold:
		# clear the gold opinions and get the empty framework
		sen_tree_map = {}
		xml_tree = ET.parse(args.gold_xml_file)
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
			# move [CLS]
			ner_tags = ''.join(pre_line[-1].split()[1:])

			if yes_not == '1':
				category = '#'.join(category_polarity[0:2]).upper()
				polarity = category_polarity[-1]
				entitys = []
				entity_list = re.finditer(entity_label, ner_tags)
				for x in entity_list:
					entitys.append(str(x.start()) + '-' + str(len(x.group())))

				# write to the xml file
				#sen_reco = ".//sentence[@id='" + sentence_id.replace("'", r"\'") + "']"
				#print(sen_reco)
				#current_sen = root.find(sen_reco)
				current_sen = sen_tree_map[sentence_id]
				current_opinions = current_sen.find('Opinions')
				if current_opinions == None:
					current_opinions = ET.Element('Opinions')
					current_sen.append(current_opinions)

				if len(entitys) == 0:	# NULL for this category&polarity
					op = ET.Element('Opinion')
					op.set('target', 'NULL')
					op.set('category', category)
					op.set('polarity', polarity)
					op.set('from', '0')
					op.set('to', '0')
					current_opinions.append(op)

				else:
					for x in entitys:
						start = int(x.split('-')[0])
						end = int(x.split('-')[1]) + start
						target_match = re.compile('\\s*'.join(sentence[start:end]))
						sentence_org = ' '.join(sentence)
						target_match_list = re.finditer(target_match, sentence_org)
						true_idx = 0
						for m in target_match_list:
							if start == sentence_org[0:m.start()].count(' '):
								break
							true_idx += 1

						gold_sentence = current_sen.find('text').text
						target_match_list = re.finditer(target_match, gold_sentence)
						match_list = []
						for m in target_match_list:
							match_list.append(str(m.start()) + '###' + str(len(m.group())) + '###' + m.group())
						if len(match_list) < true_idx + 1:
							print("Error!!!!!!!!!!!!!!!!!!!!!")
							print(len(match_list))
							print(target_match)
							print(sentence_org)
						else:
							info_list = match_list[true_idx].split('###')
							target = info_list[2]
							from_idx = info_list[0]
							to_idx = str(int(from_idx) + int(info_list[1]))
							op = ET.Element('Opinion')
							op.set('target', target)
							op.set('category', category)
							op.set('polarity', polarity)
							op.set('from', from_idx)
							op.set('to', to_idx)
							current_opinions.append(op)


		xml_string = ET.tostring(root)
		xml_write = DOM.parseString(xml_string)
		with open(args.pre_xml_file, 'w') as handle:
			xml_write.writexml(handle, indent=' ', encoding='utf-8')
