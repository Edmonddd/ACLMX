import flair
import torch
import argparse
import os
from tqdm import tqdm
import json
# os.environ['CUDA_VISIBLE-DEVICES']="0"

# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', help='Name of the output folder')
# parser.add_argument('--gpu', '-g', help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--input_file', '-if', help='Name of the test file')
# parser.add_argument('--need_consistency', help='Should we make the data consistent')
# parser.add_argument('--consistency_model', '-cm', type=str, help='Model to check the consistency of generated file')
# parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
# parser.add_argument('--gold_file_lang','-gfl',type=str,help="language of the file containing gold sentences")
# parser.add_argument('--seed','-s', type=int, help="random seed")
# parser.add_argument('--ckpt',type=str,help="checkpoint path")
# args = parser.parse_args()

#multilingual
# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', default="/home/lxm/ACLM-main/data/100",help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', default="/home/lxm/ACLM-main/data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain",help='Name of the output folder')
# parser.add_argument('--gpu', '-g', default='cuda:0',help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--input_file', '-if', default="zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain",help='Name of the test file')
# parser.add_argument('--need_consistency', default='True',help='Should we make the data consistent')
# parser.add_argument('--consistency_model', '-cm', type=str, help='Model to check the consistency of generated file')
# parser.add_argument('--file_name','-f', type=str, default="0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain", help='file name for output')
# parser.add_argument('--gold_file_lang','-gfl',type=str,default='zh',help="language of the file containing gold sentences")
# parser.add_argument('--seed','-s', type=int, default=42,help="random seed")
# parser.add_argument('--ckpt',type=str,default="/home/lxm/ACLM-main/data/100/zh_flair_xlm_100/best-model.pt",help="checkpoint path")
# args = parser.parse_args()

#multilingual_mixup
parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', default="/home/lxm/ACLMX/data/untext/ace04",help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', default="/home/lxm/ACLMX/data/untext/ace04/ace04_train-retrain-mixup-eval",help='Name of the output folder')
parser.add_argument('--gpu', '-g', default="cuda:0",help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--input_file', '-if', default="ace04_train_attn_0.3_xlm-roberta-large-ace04_train-mixup-retrain-mixup-test-mixup-t",help='Name of the test file')
parser.add_argument('--need_consistency', default='True',help='Should we make the data consistent')
parser.add_argument('--consistency_model', '-cm', type=str, help='Model to check the consistency of generated file')
parser.add_argument('--file_name','-f', type=str, default="0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-en-mixup-42-retrain", help='file name for output')
parser.add_argument('--gold_file_lang','-gfl',type=str,default='en',help="language of the file containing gold sentences")
parser.add_argument('--seed','-s', type=int, default=42,help="random seed")
parser.add_argument('--ckpt',type=str,default="/home/lxm/ACLMX/data/untext/ace04/ace04_flair/best-model.pt",help="checkpoint path")
args = parser.parse_args()

args.need_consistency = False if args.need_consistency=='False' else True

print(args)

flair.set_seed(args.seed)
torch.backends.cudnn.deterministic = True

if args.input_folder[-1]!='/':
    args.input_folder += '/'
input_folder=args.input_folder
output_folder=args.output_folder
gpu_type=args.gpu

flair.device = torch.device(gpu_type)
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
# from flair.embeddings import *
from flair.embeddings import TransformerWordEmbeddings

# https://drive.google.com/file/d/1Evf2UjlBFP2N-5jfgpOdo2uCvh53UnX3/view?usp=share_link

checkpoint = args.ckpt
test_file = args.input_file + '.txt'  
print(test_file)

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

def read_conll_file(filepath):
    # Initialize empty lists to store the sentences and gold labels
    sentences = []
    gold_labels = []
    predicted_labels = []

    # Initialize an empty list to store the current sentence
    current_sentence = []
    # Initialize an empty list to store the current gold labels
    current_gold_labels = []
    current_predicted_labels = []

    # Open the file and iterate over each line
    with open(filepath, 'r') as f:
        print('Reading the prediction file {}'.format(filepath))
        for line in tqdm(f):
            # If the line is blank, it indicates the end of a sentence
            if line.strip() == '': 
                # Add the current sentence and gold labels to the lists
                sentences.append(current_sentence)
                gold_labels.append(current_gold_labels)
                predicted_labels.append(current_predicted_labels)
                # Reset the lists for the next sentence
                current_sentence = []
                current_gold_labels = []
                current_predicted_labels = []
            else:
                # Split the line on the tab character to get the word and label
                parts = line.strip().split()
                if len(parts)==2:
                    word = ''
                else:
                    word = parts[0]
                label = parts[-2]
                predicted = parts[-1]
                # Add the word and label to the current lists
                current_sentence.append(word)
                current_gold_labels.append(label)
                current_predicted_labels.append(predicted)

    # Return the sentences and gold labels
    return sentences, gold_labels,predicted_labels

def get_undeal_sentence(sentences, predicted_labels):
    undeal = []
    print("direct processing...")
    for i in tqdm(range(len(sentences))):
        undeal.append([sentences[i],predicted_labels[i]])
    print(f"undeal.length:{len(undeal)}")
    return undeal # 返回，没有进行筛选操作的句子

def get_equal_sentences(sentences, gold_labels, predicted_labels):
    equal = []
    unequal = []
    unequal_text = []
    print("Checking for correct predictions...")
    for i in tqdm(range(len(gold_labels))):
        if args.need_consistency==False or gold_labels[i] == predicted_labels[i]:
            equal.append([sentences[i],gold_labels[i]])
        else:
            unequal.append([sentences[i],gold_labels[i]])
            unequal_text.append(sentences[i])

    print(f"equal.length:{len(equal)}")
    print(f"unequal.length:{len(unequal)}")

    return equal,unequal,unequal_text # 整个句子的gold_labels == predicted_labels 就传入equal中


def get_unequal_sentences(filepath,unequal_text):
    with open(filepath, 'w') as f:
        unequal_sentences = []
        print("produce unequal_text")
        for sentence in tqdm(unequal_text):
        # unequal_sentences_str = ' '.join(unequal_sentences)
            unequal_pair = json.dumps(sentence)
            f.write(unequal_pair + '\n')


def write_file1(filepath,undeal):
    with open(filepath,'w') as f:
        for tokens, labels in tqdm(undeal):
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

def write_file2(filepath_equal,filepath_unequal,equal,unequal):
    with open(filepath_equal,'w') as f1:
        for tokens, labels in tqdm(equal):
            for token, label in zip(tokens, labels):
                f1.write(f"{token}\t{label}\n")
            f1.write("\n")
    with open(filepath_unequal,'w') as f2:
        for tokens, labels in tqdm(unequal):
            for token, label in zip(tokens, labels):
                f2.write(f"{token}\t{label}\n")
            f2.write("\n")

def write_file(filepath, equal):
    with open(filepath, 'w') as f:
    # Iterate over the tokens and labels
        print('Writing new data (train + correct predictions) to file {}....'.format(filepath))
        # train = open(f'{data_folder}{args.gold_file_lang}_sample_train.conll','r')
        train = open(f'{data_folder}ace04_train.conll','r')
        train = train.readlines()
        for line in train:
            f.write(line)
        f.write("\n")
        for tokens, labels in tqdm(equal):
            for token, label in zip(tokens, labels):
            # Write the token and label to the file, separated by a tab character
                f.write(f"{token}\t{label}\n")
                # Add a blank line after each sentence
            f.write("\n")



if args.need_consistency == False:
    filepath = data_folder+args.input_file+'-aug+gold-test.txt'
    with open(filepath, 'w') as f:
        # train = open(f'{data_folder}{args.gold_file_lang}_sample_train.conll','r')
        train = open(f'{data_folder}ace04_train.conll','r')
        train = train.readlines()
        for line in train:
            f.write(line)
        f.write("\n")
        test = open(f'{data_folder}{test_file}','r')
        test = test.readlines()
        for line in test:
            f.write(line)
else:
    tag_type = 'ner'

    corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=test_file, dev_file=test_file, test_file=test_file, column_delimiter="\t")

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    tagger: SequenceTagger = SequenceTagger.load(checkpoint)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.final_test(output_folder,eval_mini_batch_size=256,main_evaluation_metric=('micro avg', 'f1-score'))

    sentences, gold_labels,predicted_labels = read_conll_file(output_folder + '/test.tsv') # 对新产生的句子进行遍历，[[text][gold_labels][predicted_labels]]

    # 按照原文进行筛选的句子  无误，但是不一定流畅  返回 equal , unequal
    equal,unequal,unequal_text = get_equal_sentences(sentences, gold_labels, predicted_labels) # 整个句子的gold_labels == predicted_labels
    write_file2(data_folder + '-equal.txt',data_folder + '-unequal.txt', equal, unequal)

    # 得到未筛选的句子
    undeal = get_undeal_sentence(sentences,predicted_labels)
    write_file1(data_folder+args.input_file+'-undeal.txt', undeal)



    get_unequal_sentences(data_folder+'unequal_text.json', unequal_text)

    write_file(data_folder+args.input_file+'-aug+gold.txt', equal) # 将原本的zh_sample_train.conll里面的 token label 加上equal的token label 生成到最后以-aug+gold.txt文件
