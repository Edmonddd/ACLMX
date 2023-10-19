import flair
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', default="../data/100", help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', default="../data/100/zh_flair_xlm_100", help='Name of the output folder')
parser.add_argument('--gpu', '-g', default=0,help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--train_file', '-tf', default="zh_sample_train.conll", help='train file name')
parser.add_argument('--batch_size', '-bs',default=8, type=int, help='batch-size')
parser.add_argument('--lr', '-l',default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', '-ep',default=100, type=int, help='epochs')
parser.add_argument('--language', '-lang', default='zh', help='language short code (file prefix)')
parser.add_argument('--seed', '-s', default=42, type=int, help='random seed')
args = parser.parse_args()

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
# 代码创建了一个名为 corpus 的变量，并且将其类型声明为 Corpus。用于存储一个文本语料库。 Corpus: 100 train + 800 dev + 151661 test sentences
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type) # 创建一个标签字典，用于存储文本数据中的标签信息。Dictionary with 7 tags: <unk>, LOC, CW, PROD, CORP, PER, GRP

embedding_types: List[TokenEmbeddings] = [
    # TransformerWordEmbeddings("/home/lxm/ACLM-main/xlm-roberta-large",fine_tune = True,model_max_length=256),
    TransformerWordEmbeddings("/home/hsq/ACLM-main/xlm-roberta-large",fine_tune = True,model_max_length=256),
 ] # 使用XLMRobertaModel，embedding+24层layer+pooler

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

# 这里就是100个epoch 进行lr计算，找到best-model。pt ，计算 f1-score、micro avg