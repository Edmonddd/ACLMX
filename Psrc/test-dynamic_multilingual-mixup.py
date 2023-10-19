import argparse
import transformers
import torch
import json
# parser = argparse.ArgumentParser(allow_abbrev=False)
# parser.add_argument('--model', help='which model to use')
# parser.add_argument('--input_file','-i', help='input file to use')
# parser.add_argument('--sample_generation_mode', help='static/dynamic generation')
# parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
# parser.add_argument('--mask_entities', default='False', help='Should we mask the entities following the gaussian distribution')
# parser.add_argument('-a', '--mask_attn', default='all', help='Which attn mode to select (all/gauss/none)')
# parser.add_argument('--mode','-m', default='attn', help='masking mode (both/either/attn)')
# parser.add_argument('--topk', default='50', help='topk value')
# parser.add_argument('--num_of_sequences', default=5, help='number of sequences per datapoint')
# parser.add_argument('--max_length', default=100, help='max_length of the generated sentence')
# parser.add_argument('--do_sample', default='True', help='do_sample argument')
# parser.add_argument('--num_beams', default=5, help='num_beams argument')
# parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
# parser.add_argument('--root_dir','-ro', type=str, default='', help='root directory')
# parser.add_argument('--lang','-la', type=str, default='', help='language you are working on')
# parser.add_argument('--remove_repetitions', default='True', help="should remove repetitions?")
# parser.add_argument('--seed', '-s', type=int, default=-1, help="random state seed")
# args = parser.parse_args()

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--model', default="/home/lxm/ACLMX/data/untext/ace04/ace04_train_attn_0.3_xlm-roberta-large-ace04-mixup-bart-retrain-final",help='which model to use')
parser.add_argument('--input_file','-i', default="ace04_train_attn_0.3_xlm-roberta-large",help='input file to use')
parser.add_argument('--sample_generation_mode', default='dynamic',help='static/dynamic generation')
parser.add_argument('--directory','-dir', default="/home/lxm/ACLMX/data/untext/ace04", help='data directory where train and dev files are located')
parser.add_argument('--mask_entities', default='False', help='Should we mask the entities following the gaussian distribution')
parser.add_argument('-a', '--mask_attn', default='gauss', help='Which attn mode to select (all/gauss/none)')
parser.add_argument('--mode','-m', default='attn', help='masking mode (both/either/attn)')
parser.add_argument('--topk', default=10, help='topk value')
parser.add_argument('--num_of_sequences', default=5, help='number of sequences per datapoint')
parser.add_argument('--max_length', default=120, help='max_length of the generated sentence')  # ace04 untext 句子最大长度
# parser.add_argument('--max_length', default=100, help='max_length of the generated sentence')  ace05 untext 句子最大长度
parser.add_argument('--do_sample', default='True', help='do_sample argument')
parser.add_argument('--num_beams', default=5, help='num_beams argument')
parser.add_argument('--file_name','-f', type=str, default="ace04_train-mixup-retrain-mixup-test", help='file name for output')
parser.add_argument('--root_dir','-ro', type=str, default="/home/lxm/ACLMX/data/untext/ace04", help='root directory')
parser.add_argument('--lang','-la', type=str, default="en_XX", help='language you are working on')
parser.add_argument('--remove_repetitions', default='False', help="should remove repetitions?")
parser.add_argument('--seed', '-s', type=int, default=42, help="random state seed")
args = parser.parse_args()

args.remove_repetitions = False if args.remove_repetitions=='False' else True
args.mask_entities = False if args.mask_entities=='False' else True
args.do_sample = False if args.do_sample=='False' else True

print(args)

if not args.seed==-1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

from transformers import pipeline
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
tokenizer = MBart50Tokenizer.from_pretrained(args.model)
tokenizer.src_lang = args.lang
model_pipeline = pipeline("text2text-generation", model=args.model, tokenizer=tokenizer, device=0)
import os
import pandas as pd
import random
import sys
from utils import mask_entities, mask_words,get_random_gauss_value
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


if args.directory[-1]!='/':
    args.directory += '/'

newFileName = args.directory + args.input_file + '.csv'

data = pd.read_csv(newFileName)
label = list(data.label.values)
text = list(data.text.values)
bert_att = list(data.bert_att.values)

new_tokens = ['<b-org>','<b-gpe>', '<b-per>', '<b-loc>', '<b-fac>', '<b-veh>', '<b-wea>', 
                '<b-org-org>','<b-org-gpe>','<b-org-per>', '<b-org-loc>', '<b-org-fac>', '<b-org-veh>', '<b-org-wea>',
                '<b-gpe-org>','<b-gpe-gpe>','<b-gpe-per>', '<b-gpe-loc>', '<b-gpe-fac>', '<b-gpe-veh>', '<b-gpe-wea>', 
                '<b-per-org>','<b-per-gpe>','<b-per-per>', '<b-per-loc>', '<b-per-fac>', '<b-per-veh>', '<b-per-wea>',
                '<b-loc-org>','<b-loc-gpe>','<b-loc-per>', '<b-loc-loc>', '<b-loc-fac>', '<b-loc-veh>', '<b-loc-wea>',
                '<b-fac-org>','<b-fac-gpe>','<b-fac-per>', '<b-fac-loc>', '<b-fac-fac>', '<b-fac-veh>', '<b-fac-wea>',
                '<b-veh-org>','<b-veh-gpe>','<b-veh-per>', '<b-veh-loc>', '<b-veh-fac>', '<b-veh-veh>', '<b-veh-wea>',
                '<b-wea-org>','<b-wea-gpe>','<b-wea-per>', '<b-wea-loc>', '<b-wea-fac>', '<b-wea-veh>', '<b-wea-wea>',
                '<i-org>','<i-gpe>', '<i-per>', '<i-loc>', '<i-fac>', '<i-veh>', '<i-wea>',
                '<i-org-org>','<i-org-gpe>','<i-org-per>', '<i-org-loc>', '<i-org-fac>', '<i-org-veh>', '<i-org-wea>',
                '<i-gpe-org>','<i-gpe-gpe>','<i-gpe-per>', '<i-gpe-loc>', '<i-gpe-fac>', '<i-gpe-veh>', '<i-gpe-wea>', 
                '<i-per-org>','<i-per-gpe>','<i-per-per>', '<i-per-loc>', '<i-per-fac>', '<i-per-veh>', '<i-per-wea>',
                '<i-loc-org>','<i-loc-gpe>','<i-loc-per>', '<i-loc-loc>', '<i-loc-fac>', '<i-loc-veh>', '<i-loc-wea>',
                '<i-fac-org>','<i-fac-gpe>','<i-fac-per>', '<i-fac-loc>', '<i-fac-fac>', '<i-fac-veh>', '<i-fac-wea>',
                '<i-veh-org>','<i-veh-gpe>','<i-veh-per>', '<i-veh-loc>', '<i-veh-fac>', '<i-veh-veh>', '<i-veh-wea>',
                '<i-wea-org>','<i-wea-gpe>','<i-wea-per>', '<i-wea-loc>', '<i-wea-fac>', '<i-wea-veh>', '<i-wea-wea>',
                ]
new_tokens = set(new_tokens)


print('calculate similar sentences') # 计算相似句子
model = SentenceTransformer("/mnt/nas/lxm/Project/ACLMX/distiluse-base-multilingual-cased-v2")
embeddings = model.encode(text, show_progress_bar=True, batch_size=1024) # text是train文件

text_similar = [] # 得到 text_similar
for i in range(len(embeddings)):
    arr = []
    for j in range(len(embeddings)):
        arr.append(1 - cosine(embeddings[i], embeddings[j]))  # 再次从text中遍历每个句子，计算每个句子和i句子语义相似句子

    top_2 = sorted(range(len(arr)), key=lambda i: arr[i])[-3:-1] # 取倒数第3和倒数第2大的元素的索引，top_2 中包含的是列表 arr 中最大的两个元素的索引
    text_similar.append(top_2)

del model, embeddings
print('done')

def remove_tags(temp):
    return ' '.join([i for i in temp.split() if i[0]!='<'])

def isGeneratedSentenceValid(sent):
    global new_tokens

    count = 0
    for i in sent.split(' '):
        if i!='':
            if (i[0]=='<' and i[-1]!='>') or (i[0]!='<' and i[-1]=='>'):
                return False

            if i[0]=='<' and i[-1]=='>':
                if not i in new_tokens:
                    return False
                count+=1
    if count%2:
        return False

    return True

# args.model[:-6].strip(args.file_name) + '-' +
generated_file = args.root_dir + "/" + args.input_file + '-' + args.file_name + '-mixup-t' + '.txt'
if os.path.exists(generated_file):
    os.remove(generated_file)
print(generated_file)


# 新增增强文件用以记录
augment_file_1 = args.root_dir + "/" + args.input_file + '-' + args.file_name + '-test' + '.txt'
if os.path.exists(augment_file_1):
    os.remove(augment_file_1)
print(augment_file_1)

# augment_file_2 = args.root_dir + "/" + args.input_file + '-' + args.file_name + '-test2' + '.txt'
# if os.path.exists(augment_file_2):
#     os.remove(augment_file_2)
# print(augment_file_2)

# DYNAMIC MASKING NEW CODE
if args.sample_generation_mode=='static':
    with open(generated_file, 'a') as the_file:
        test = 0
        for i in tqdm(range(len(text))):
            saved = {}
            new_text = text[i].split()
            new_label = label[i].split()
            new_bert_attn = bert_att[i].split()
            if args.mode == 'attn':
                new_sketch = mask_entities(new_text, new_label, False)
                new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
            elif args.mode == 'both':
                new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
            elif args.mode == 'either':
                temp = get_random_gauss_value(0.5,0.3)
                if temp<=0.5:
                    new_sketch = mask_entities(new_text, new_label, False)
                    new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                else:
                    new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                    new_sketch = mask_words(new_sketch, new_label, new_bert_attn, 'none')
            # the_file.write('Original: ' + text[i] + '\n')
            generated_text = model_pipeline(new_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length), num_return_sequences=int(args.num_of_sequences), forced_bos_token_id=tokenizer.lang_code_to_id[args.lang])
            for z in range(int(args.num_of_sequences)):
                if args.remove_repetitions:
                    if generated_text[z]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[z]['generated_text']] = 1

                if not isGeneratedSentenceValid(generated_text[z]['generated_text']):
                    test+=1
                    continue

                # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                prev_label = ''
                temp = False
                for k in generated_text[z]['generated_text'].split(' '):
                    if k=='':
                        continue
                    if prev_label=='' and k[0]!='<':
                        the_file.write(f'{k}\tO\n')
                    elif prev_label!='' and k[0]=='<':
                        the_file.write(f'\t{prev_label}\n')
                        prev_label=''
                        temp = False
                        continue
                    elif k[0]=='<':
                        prev_label = k[1:-1].upper()
                        continue
                    else:
                        if temp:
                            the_file.write(f' {k}')
                        else:
                            temp = True
                            the_file.write(f'{k}')

                the_file.write('\n')
            # the_file.write('\n')
elif args.sample_generation_mode=='dynamic':
    with open(generated_file, 'a') as the_file:
        with open(augment_file_1,'a') as fir_file:
        #with open(augment_file_2,'a') as sec_file:
            test = 0
            for i in tqdm(range(len(text))):
                saved = {}
                for z in range(int(args.num_of_sequences)-2): # 这里是对原先句子进行操作，生成3个新的含关键字和标签的句子
                    new_text = text[i].split()
                    new_label = label[i].split()
                    new_bert_attn = bert_att[i].split()

                    if args.mode == 'attn':
                        new_sketch = mask_entities(new_text, new_label, False)
                        new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                    elif args.mode == 'both':
                        new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                        new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                    elif args.mode == 'either':
                        temp = get_random_gauss_value(0.5,0.3)
                        if temp<=0.5:
                            new_sketch = mask_entities(new_text, new_label, False)
                            new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                        else:
                            new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                            new_sketch = mask_words(new_sketch, new_label, new_bert_attn, 'none')

                    generated_text = model_pipeline(new_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length)) # 通过关键字和实体标签生成新的句子，赋值给 generated_text
                    if generated_text[0]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[0]['generated_text']] = 1

                    if not isGeneratedSentenceValid(generated_text[0]['generated_text']):
                        test+=1
                        continue

                    # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                    # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                    prev_label = ''
                    temp = False
                    generate_sentence = []
                    for k in generated_text[0]['generated_text'].split(' '):
                        if k=='':
                            continue

                        # generate_sentence.append(k)

                        if prev_label=='' and k[0]!='<':
                            the_file.write(f'{k}\tO\n')
                            # fir_file.write(f'{k}\tO\n')
                            generate_sentence.append(k)
                        elif prev_label!='' and k[0]=='<':
                            the_file.write(f'\t{prev_label}\n')
                            # generate_sentence.append(k)
                            # fir_file.write(f'\t{prev_label}\n')
                            prev_label=''
                            temp = False
                            continue
                        elif k[0]=='<':
                            prev_label = k[1:-1].upper()
                            continue
                        else:
                            if temp:
                                the_file.write(f' {k}')
                                generate_sentence.append(k)
                                # fir_file.write(f' {k}')
                            else:
                                temp = True
                                the_file.write(f'{k}')
                                generate_sentence.append(k)
                                # fir_file.write(f'{k}')
                    generate_sentence_str = ' '.join(generate_sentence)
                    sentences_pair = ["start: " + text[i],"asdfg: " + generate_sentence_str]
                    sentences_pair = json.dumps(sentences_pair)
                    fir_file.write(sentences_pair + '\n')
                    
                    # fir_file.write('\n')
                    the_file.write('\n')

                for z in range(2): # 这里是对相似句子进行操作，生成2个句子
                    new_text = text[i].split()
                    new_label = label[i].split()
                    new_bert_attn = bert_att[i].split()

                    text_similar_index = text_similar[i][z] 
                    new_text += text[text_similar_index].split()
                    new_label += label[text_similar_index].split()
                    new_bert_attn += bert_att[text_similar_index].split()

                    if args.mode == 'attn':
                        new_sketch = mask_entities(new_text, new_label, False)
                        new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                    elif args.mode == 'both':
                        new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                        new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                    elif args.mode == 'either':
                        temp = get_random_gauss_value(0.5,0.3)
                        if temp<=0.5:
                            new_sketch = mask_entities(new_text, new_label, False)
                            new_sketch = mask_words(new_sketch, new_label, new_bert_attn, args.mask_attn)
                        else:
                            new_sketch = mask_entities(new_text, new_label, args.mask_entities)
                            new_sketch = mask_words(new_sketch, new_label, new_bert_attn, 'none')

                    generated_text = model_pipeline(new_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length))
                    if generated_text[0]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[0]['generated_text']] = 1

                    if not isGeneratedSentenceValid(generated_text[0]['generated_text']):
                        test+=1
                        continue

                    # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                    # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                    prev_label = ''
                    temp = False
                    generate_sentence = []
                    for k in generated_text[0]['generated_text'].split(' '):
                        if k=='':
                            continue
                        if prev_label=='' and k[0]!='<':
                            the_file.write(f'{k}\tO\n')
                            generate_sentence.append(k)
                            # Sec_file.write(f'{k}\tO\n')
                        elif prev_label!='' and k[0]=='<':
                            the_file.write(f'\t{prev_label}\n')
                            # generate_sentence.append(k)
                            # Sec_file.write(f'\t{prev_label}\n')
                            prev_label=''
                            temp = False
                            continue
                        elif k[0]=='<':
                            prev_label = k[1:-1].upper()
                            continue
                        else:
                            if temp:
                                the_file.write(f' {k}')
                                generate_sentence.append(k)
                                # Sec_file.write(f' {k}')
                            else:
                                temp = True
                                the_file.write(f'{k}')
                                generate_sentence.append(k)
                                # Sec_file.write(f'{k}')
                    # Sec_file.write('\n')
                    generate_sentence_str = ' '.join(generate_sentence)
                    sentences_pair = ["start: " + text[i], "asdfgh: " + generate_sentence_str]
                    sentences_pair = json.dumps(sentences_pair)
                    fir_file.write(sentences_pair + '\n')
                    # sec_file.write(sentences_pair + '\n')


                    the_file.write('\n')

                # the_file.write('----\n\n')
                # the_file.write('\n')


print('File generated at: ', generated_file)