# coding=utf-8

"""Processors for Semeval Dataset."""

import csv
import os
import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, ner_labels_a=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            ner_labels_a: ner tag sequence for text_a. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.ner_labels_a = ner_labels_a


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_ner_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Semeval_Processor(DataProcessor):
    """Processor for the SemEval 2015 and 2016 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "train_TAS.tsv"), 'r', encoding='utf-8') as fin:
            fin.readline()
            train_data = fin.readlines()
            return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "dev_TAS.tsv"), 'r', encoding='utf-8') as fin:
            fin.readline()
            dev_data = fin.readlines()
            return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "test_TAS.tsv"), 'r', encoding='utf-8') as fin:
            fin.readline()
            test_data = fin.readlines()
            return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def get_ner_labels(self, data_dir):
        ner_labels = ['[PAD]', '[CLS]']
        with open(os.path.join(data_dir, "train_TAS.tsv"), 'r', encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                tags = line.strip().split('\t')[-1].split()
                for x in tags:
                    if x not in ner_labels:
                        ner_labels.append(x)
        print(ner_labels)
        return ner_labels

    def _create_examples(self, lines, set_type):
        """Creates examples."""
        examples = []
        for (i, line) in enumerate(lines):
            line_arr = line.strip().split('\t')
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line_arr[3])) # sentence
            text_b = tokenization.convert_to_unicode(str(line_arr[2])) # category_polarity
            label = tokenization.convert_to_unicode(str(line_arr[1])) # yes or no
            ner_labels_a = tokenization.convert_to_unicode(str(line_arr[4])) # ner tags

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, ner_labels_a=ner_labels_a))
        return examples
