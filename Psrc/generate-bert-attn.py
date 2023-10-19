import torch
from transformers import AutoModel,AutoTokenizer
import numpy as np
import re
from tqdm import tqdm
import sys
import random
import argparse
from stopwordsiso import stopwords
import pandas as pd

import os


stopwords_all = stopwords(["bn","de", "es","en","hi","ko","nl","ru","tr","zh"])

parser = argparse.ArgumentParser()
# parser.add_argument('-a', type=str, required=True, help="mode attn/random")
# parser.add_argument('-m', type=float, required=True, help="masking rate")
# parser.add_argument('-dir', type=str, required=True, help="directory to save everything")
# parser.add_argument('-ckpt', type=str, required=True, help="model file path")
# parser.add_argument('-tf', type=str, required=True, help="training file path")
# parser.add_argument('-df', type=str, required=True, help="dev file path")

parser.add_argument('-a', default='attn', type=str,  help="mode attn/random")
parser.add_argument('-m', default=0.3, type=float,  help="masking rate")
parser.add_argument('-dir',default="/mnt/nas/lxm/Project/ACLMX/data/ace2005",  type=str, help="directory to save everything")
parser.add_argument('-ckpt',default="/mnt/nas/lxm/Project/ACLMX/data/ace2005/ace05_flair/best-model.pt",  type=str,  help="model file path")
parser.add_argument('-tf',default="/mnt/nas/lxm/Project/ACLMX/data/ace2005/ace05_train_context.json",  type=str,  help="training file path")
parser.add_argument('-df',default="/mnt/nas/lxm/Project/ACLMX/data/ace2005/ace05_dev_context.json",  type=str,  help="dev file path")

args = parser.parse_args()

mode = args.a
k = args.m
dir = args.dir

model_string = 'xlm-roberta-large'
checkpoint = args.ckpt
model = AutoModel.from_pretrained("/mnt/nas/lxm/Project/ACLMX/xlm-roberta-large",output_hidden_states=True, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/nas/lxm/Project/ACLMX/xlm-roberta-large", add_prefix_space=True)
bert_keys = list(model.state_dict().keys())
print(len(bert_keys))

ckpt = torch.load(checkpoint) # 395条

try:
    ckpt_keys = list(ckpt['state_dict'].keys())

    print(len(ckpt_keys))
    count = 0

    for i in range(len(bert_keys)):

        if bert_keys[i] in ckpt_keys[i]:

            ckpt['state_dict'][bert_keys[i]] = ckpt['state_dict'][ckpt_keys[i]]
            count += 1

    if mode=='attn':
        model.load_state_dict(ckpt['state_dict'], strict = False)
except:
    ckpt_keys = list(ckpt.keys())

    print(len(ckpt_keys))
    count = 0

    for i in range(len(bert_keys)):

        if bert_keys[i] in ckpt_keys[i]:

            ckpt[bert_keys[i]] = ckpt[ckpt_keys[i]]


            count += 1

    if mode=='attn':
        model.load_state_dict(ckpt, strict = False)


def getAttentionMatrix(sent):
    tokens = [] 

    sentence_split = sent.split(" ")

    x = tokenizer(sentence_split, return_tensors='pt',is_split_into_words=True)
    # ***
    word_ids = x.word_ids()

    toDelete = []

    for i in range(len(word_ids)):
        id = word_ids[i]
        if id==None:
            toDelete = toDelete + [i]
            continue

        id = id-1

        if len(tokens)==id:
            tokens = tokens + [[i]]
        else:
            tokens[id].append(i)

    output = model(**x)

    attention_matrix = np.zeros([len(tokens), len(tokens)]) # 30*30的0注意力矩阵。

    for i in range(len(tokens)):
            toDelete = toDelete + tokens[i][1:]

    for layer in range(8,12): # Keyword Selection 中得到注意力得分，取每个头和最后4层的注意力得分相加

        attention = output[3][layer][0].detach().numpy()

        attention = np.sum(attention, axis= 0) # 计算某一层的注意力输出的总和或平均值。最终，attention 变量将包含对某一层的注意力输出的汇总信息，可以用于进一步的分析或可视化。

        for i in range(len(tokens)):

            if(len(tokens[i])>1):
                attention[:,tokens[i][0]] = np.sum(attention[:, tokens[i][0]:tokens[i][0]+len(tokens[i])], axis=1) # 这行代码计算了注意力矩阵的列（对应不同位置或标记）的总和，并将结果放回 attention 矩阵的相应列。将一组连续的注意力分数合并为单个分数。

        for i in range(len(tokens)):

            if(len(tokens[i])>1):
                attention[tokens[i][0],:] = np.mean(attention[tokens[i][0]:tokens[i][0]+len(tokens[i]),:], axis =0) # 这行代码计算了注意力矩阵的行（对应不同位置或标记）的均值，并将结果放回 attention 矩阵的相应行。将一组连续的注意力分数合并为单个分数。

        attention = np.delete(attention,toDelete, axis=1)
        attention = np.delete(attention,toDelete, axis=0)


        attention_matrix = np.add(attention_matrix, attention)


    return attention_matrix # 合并和删除一些分数，最终将它们添加到另一个矩阵中（attention_matrix）。得到注意力矩阵


def getMasksTry(sent,k):
    try:
        x,y = getMasks(sent,k)
        return x,y # x 为含['词','类型','是否unmasked'] , y 为unmasked的列表['ners','topK']。
    except:

        ignore = []
        ners = []
        for i in range(len(sent)):
            t = sent[i]

            if t[1].startswith('B'):
                ners.append([i])
            elif t[1].startswith('I'):
                ignore.append(i)
                ners[-1].append(i)
            elif re.search(r'\W_', t[0]) or t[0] in stopwords_all:
                ignore.append(i)

        unmasked = []

        new_input = []
        for t in sent:
            new_input.append([t[0],t[1],0])

        for i in range(len(ignore)):
            new_input[ignore[i]][2] =0

        for i in range(len(ners)):
            for t in ners[i]:
                new_input[t][2] =1

        return new_input, unmasked


def getMasks(sent, k=0.2):

    countners=0

    sequence = ''
    ignore = []
    ners = []
    for i in range(len(sent)):
        t = sent[i]

        sequence = sequence+" "+t[0]

        if t[1].startswith('B'):
            ners.append([i])
            countners=countners+1
        elif t[1].startswith('I'):
            ignore.append(i)
            ners[-1].append(i)
            countners=countners+1
        elif re.search(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', t[0])  or t[0] in stopwords_all:
            ignore.append(i)
        # 如果当前单词不属于命名实体或是特定字符（通过正则表达式或停用词检查），将其索引 i 添加到 ignore 列表中。 ner=实体类型的索引组，ignore=I类型的索引组+特殊字符和stopwords里面的词

    k = int(np.ceil(k*(len(sent)-countners))) # 向上取整
    k = min(k, len(sent))

    attentionMatrix = getAttentionMatrix(sequence)



    for i in range(len(ners)):
        attentionMatrix[:,ners[i][0]] = np.sum(attentionMatrix[:, ners[i][0]:ners[i][0]+len(ners[i])], axis=1)



    for i in range(len(ners)):
        attentionMatrix[ners[i][0],:] = np.mean(attentionMatrix[ners[i][0]:ners[i][0]+len(ners[i]),:], axis =0)
    # 对 attentionMatrix 矩阵中与实体相关的部分进行一些操作，包括合并一些分数
    unmasked = []

    for i in range(len(attentionMatrix)):
        attentionMatrix[i][i] = -100 # 自身对自身的注意力变成-100

    attentionMatrix[:,ignore]= -100 # ignore 词的注意力赋值为-100

    modelInput = "[MASK]"

    new_input = []

    for i in range(len(ners)):
        for t in ners[i]:
            unmasked.append(t) # ners 中的词加入到 unmasked里面。

        topK = np.argpartition(attentionMatrix[ners[i][0]], -k)[-k:] # 找到topK个注意力前面的关键字

        for t in topK:
            unmasked.append(t) # topK 个词加入到 unmasked里面

    unmasked = list(set(unmasked)) # 转化为unmasked 列表 unmasked里面有ners 和 topK注意力的词

    unmasked.sort()

    for t in sent:
        new_input.append([t[0],t[1],0]) # new_input：[['考','O','0']]

    for i in range(len(unmasked)):
        new_input[unmasked[i]][2] =1 # new_input中第二列按照是否在unmasked中 变成1.

    for i in range(len(ignore)):
        new_input[ignore[i]][2] =0 # 同上，将属于ignore 中的词变成0

    for i in range(len(ners)):
        for t in ners[i]:
            new_input[t][2] =1

    return new_input, unmasked 

def getMasks2(sent, k=0.2):

    countners=0

    sequence = ''
    ignore = []
    ners = []
    indexes_to_consider = []
    for i in range(len(sent)):
        t = sent[i]

        sequence = sequence+" "+t[0]

        if t[1].startswith('B'):
            ners.append([i])
            countners=countners+1
        elif t[1].startswith('I'):
            ignore.append(i)
            ners[-1].append(i)
            countners=countners+1
        elif re.search(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', t[0]):
            ignore.append(i)
        else:
            indexes_to_consider.append(i)

    k = int(np.ceil(k*len(indexes_to_consider)))
    k = min(k, len(sent))
    unmasked = random.sample(indexes_to_consider,k)
    new_input = []

    for t in sent:
        new_input.append([t[0],t[1],0])

    for i in range(len(unmasked)):
        new_input[unmasked[i]][2] =1

    for i in range(len(ignore)):
        new_input[ignore[i]][2] =0

    for i in range(len(ners)):
        for t in ners[i]:
            new_input[t][2] =1

    return new_input, unmasked

def split_at_punctuation(s):
    delimiters = re.findall(r',', s)
    split_list = re.split(r',', s)
    result = []
    for i, item in enumerate(split_list):
        result.append(item)
        if i < len(split_list) - 1:
            result.append(delimiters[i])

    return_result = [string for string in result if (len(string) > 0 and string!='̇')]
    return return_result
# 这段代码的主要功能是将输入的字符串按照逗号分割，并保留逗号作为分隔符，同时去除空字符串和特定字符。
def process_file(file_path):
  # Open the file
  with open(file_path) as file:
    # Create an empty list to store the results
    result = []

    # Create an empty list to store the current data
    current_list = []

    # Read each line of the file
    for line in file:
      # Remove the newline character from the line
      line = line.strip('\n')

      # Split the line by the tab character and store it in a tuple
      line_tuple = tuple(line.split('\t'))


      # Check if the tuple contains an empty string
      if not '' in line_tuple:
        # Append the tuple to the current list
        current_list.append(line_tuple)



      # If the line is blank, append the current list to the result list
      # if it's not empty, and start a new list
      if not line.strip():
        if current_list:
          result.append(current_list)
          current_list = []

    # Append the current list to the result list, in case there was no
    # blank line at the end of the file
    if current_list:
      result.append(current_list)

    # Return the result list
    return result



def process_file2(file_path):
  # Open the file
  with open(file_path) as file:
    # Create an empty list to store the results
    result = []

    # Create an empty list to store the current data
    current_list = []

    # Read each line of the file
    for line in file:
      # Remove the newline character from the line
        line = line.strip('\n')

        if not line.strip():
            if current_list:
                result.append(current_list)
                current_list = []
                continue

        thisLine = line.split('\t')

        # line_tuple = tuple(line.split('\t'))

        words = split_at_punctuation(thisLine[0])

      # Split the line by the tab character and store it in a tuple

        for i in range(len(words)):
            word = words[i]

            if i==0:
                line_tuple = tuple([word,thisLine[1]])
            elif thisLine[1]=='O':
                line_tuple = tuple([word,thisLine[1]])
            else:
                line_tuple = tuple([word, str('I'+thisLine[1][1:])])


            # Check if the tuple contains an empty string
            if not '' in line_tuple:
                # Append the tuple to the current list
                current_list.append(line_tuple)



      # If the line is blank, append the current list to the result list
      # if it's not empty, and start a new list


    # Append the current list to the result list, in case there was no
    # blank line at the end of the file
    if current_list:
        result.append(current_list)

    return result


def generateMasks(path, out,k, mode):
    sentences = process_file2(path) # 生成的一个一个句子，每个句子格式为：[('考','O'),...]
    newsents = []
    unmasked = []
    for sent in tqdm(sentences):
        if mode=="attn" or mode=="plm" :
            newsent, unmasks= getMasksTry(sent,k)
        elif mode=="random":
            newsent,unmasks = getMasks2(sent,k)
        else:
            raise Exception("Not valid mode")

        newsents.append(newsent)
        unmasked.append(unmasks)

    with open(out,"w") as f:
        for s in newsents:
            for token in s:
                f.write(token[0]+'\t'+token[1]+'\t'+str(token[2])+"\n")
            f.write("\n")

    return newsents,unmasked

if dir[-1] != '/':
    dir += '/'


print(mode,k,dir)

train_file = args.tf.split('/')[-1].split('.')[0]
dev_file = args.df.split('/')[-1].split('.')[0]

train_file_name = dir+train_file+"_"+mode+"_"+str(k)+"_"+model_string
dev_file_name = dir+dev_file+"_"+mode+"_"+str(k)+"_"+model_string

a,b= generateMasks(args.tf, train_file_name+".txt",k, mode) # 对train_file 文件进行生成Masks 
a,b= generateMasks(args.df, dev_file_name+".txt",k, mode) # 对dev_file 文件进行生成Masks  


def sketch9 (data):
    '''
       Saving normal text, corresponding label and bert att 
    '''
    text = ''
    label_sent = ''
    bert_sent = ''
    final = []
    for i in tqdm(data):
        if i == '':
            if text!='':
                final.append([text.strip(),label_sent.strip(), bert_sent.strip()])
            text = ''
            label_sent = ''
            bert_sent = ''
            continue

        word, label, bert_att = i.split('\t')

        bert_sent += ' ' + bert_att
        text += ' ' + word
        label_sent += ' ' + label

    dataset = pd.DataFrame(final, columns=['text', 'label', 'bert_att']) # 创建了一个名为 dataset 的 Pd DataFrame 对象，其中包含三列数据
    return dataset

with open(train_file_name+".txt", 'r') as f:
    data = f.read().splitlines()

dataset = sketch9(data)
dataset.to_csv(train_file_name+".csv", index=False)

with open(dev_file_name+".txt", 'r') as f:
    data = f.read().splitlines()

dataset = sketch9(data)
dataset.to_csv(dev_file_name+".csv", index=False)