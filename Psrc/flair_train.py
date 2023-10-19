import flair
import torch
import argparse
import os

# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', help='Name of the output folder')
# parser.add_argument('--gpu', '-g', help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--train_file', '-tf', help='train file name')
# parser.add_argument('--batch_size', '-bs',type=int, help='batch-size')
# parser.add_argument('--lr', '-l',type=float, help='learning rate')
# parser.add_argument('--epochs', '-ep',type=int, help='epochs')
# parser.add_argument('--language', '-lang', help='language short code (file prefix)')
# parser.add_argument('--seed', '-s', type=int, help='random seed')
# args = parser.parse_args()

# multilingual第一次的train 
# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', default="/home/lxm/ACLM-main/data/100",help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', default="/home/lxm/ACLM-main/data/100/zh_flair_xlm_100",help='Name of the output folder')
# parser.add_argument('--gpu', '-g', default='cuda:0',help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--train_file', '-tf', default="zh_sample_train.conll",help='train file name')
# parser.add_argument('--batch_size', '-bs',type=int, default=8,help='batch-size')
# parser.add_argument('--lr', '-l',type=float, default=0.01,help='learning rate')
# parser.add_argument('--epochs', '-ep',type=int, default=100,help='epochs')
# parser.add_argument('--language', '-lang', default='zh',help='language short code (file prefix)')
# parser.add_argument('--seed', '-s', type=int, default=42,help='random seed')
# args = parser.parse_args()

# multilingual第二次的train
# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', default="/home/lxm/ACLM-main/data/100",help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', default="/home/lxm/ACLM-main/data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-flair",help='Name of the output folder')
# parser.add_argument('--gpu', '-g', default='cuda:0',help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--train_file', '-tf', default="zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-aug+gold.txt",help='train file name')
# parser.add_argument('--batch_size', '-bs',type=int, default=8,help='batch-size')
# parser.add_argument('--lr', '-l',type=float, default=0.01,help='learning rate')
# parser.add_argument('--epochs', '-ep',type=int, default=100,help='epochs')
# parser.add_argument('--language', '-lang', default='zh',help='language short code (file prefix)')
# parser.add_argument('--seed', '-s', type=int, default=42,help='random seed')
# args = parser.parse_args()

# multilingual_mixup第一次的train    untext
parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', default="/mnt/nas/lxm/Project/ACLMX/data/ace2005",help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', default="/mnt/nas/lxm/Project/ACLMX/data/ace2005/ace05_flair",help='Name of the output folder')
parser.add_argument('--gpu', '-g', default='cuda',help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--train_file', '-tf', default="data/ace2005/ace05_train_context.json",help='train file name')
parser.add_argument('--batch_size', '-bs',type=int, default=8,help='batch-size')
parser.add_argument('--lr', '-l',type=float, default=0.01,help='learning rate')
parser.add_argument('--epochs', '-ep',type=int, default=100,help='epochs')
parser.add_argument('--language', '-lang', default='ace05',help='language short code (file prefix)')
parser.add_argument('--seed', '-s', type=int, default=42,help='random seed')
args = parser.parse_args()

# multilingual_mixup第一次的train   text
# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', default="/home/hsq/ACLM-main/data/untext/ace04",help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', default="/home/hsq/ACLM-main/data/untext/ace04/ace04_flair_xlm_100",help='Name of the output folder')
# parser.add_argument('--gpu', '-g', default='cuda:0',help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--train_file', '-tf', default="ace04_train.conll",help='train file name')
# parser.add_argument('--batch_size', '-bs',type=int, default=8,help='batch-size')
# parser.add_argument('--lr', '-l',type=float, default=0.01,help='learning rate')
# parser.add_argument('--epochs', '-ep',type=int, default=100,help='epochs')
# parser.add_argument('--language', '-lang', default='ace04',help='language short code (file prefix)')
# parser.add_argument('--seed', '-s', type=int, default=42,help='random seed')
# args = parser.parse_args()


# multilingual_mixup第二次的train 
# parser = argparse.ArgumentParser(description='Train flair model')
# parser.add_argument('--input_folder', '-i', default="/home/hsq/ACLM-main/data/100",help='Name of the input folder containing train, dev and test files')
# parser.add_argument('--output_folder', '-o', default="/home/hsq/ACLM-main/data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-mixup-42-retrain-mixup-flair",help='Name of the output folder')
# parser.add_argument('--gpu', '-g', default='cuda:0',help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
# parser.add_argument('--train_file', '-tf', default="zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-mixup-42-retrain-mixup-aug+gold.txt",help='train file name')
# parser.add_argument('--batch_size', '-bs',type=int, default=8,help='batch-size')
# parser.add_argument('--lr', '-l',type=float, default=0.01,help='learning rate')
# parser.add_argument('--epochs', '-ep',type=int, default=100,help='epochs')
# parser.add_argument('--language', '-lang', default='zh',help='language short code (file prefix)')
# parser.add_argument('--seed', '-s', type=int, default=42,help='random seed')
# args = parser.parse_args()

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
from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=args.train_file,
                              dev_file=f'{args.language}_dev.conll',test_file=f'{args.language}_test.conll',column_delimiter="\t", comment_symbol="# id")

tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    TransformerWordEmbeddings("/mnt/nas/lxm/Project/ACLMX/xlm-roberta-large",fine_tune = True,model_max_length=256),
 ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger: SequenceTagger = SequenceTagger(use_rnn = False,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train(output_folder, learning_rate=args.lr,save_final_model=False,
             mini_batch_size=args.batch_size,
             max_epochs=args.epochs,embeddings_storage_mode='gpu',main_evaluation_metric=('micro avg', 'f1-score'), shuffle=True)