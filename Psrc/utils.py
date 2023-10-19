import random
random.seed(5)

def get_random_gauss_value(mean, std_dev):
    mask_rate = -1

    while mask_rate < 0 or mask_rate > 1:
        mask_rate = random.gauss(mean, std_dev)

    return mask_rate # 得到guass分布采样率


def find_masks_for_entities(start_index, end_index):

    mask_rate = get_random_gauss_value(0.5, 1/(end_index-start_index+1))
    mask_count = round((end_index-start_index+1)*mask_rate)

    return random.sample(list(range(start_index,end_index+1)), mask_count)

def merge_list(text):
    final_text = '      '
    for i in range(len(text)):
        if text[i] == '<mask>':
            if final_text[-6:] == '<mask>':
                continue
            else:
                final_text += ' <mask>'
        else:
            final_text += ' ' + text[i]

    # if final_text[-6:] != '<mask>' and final_text[-1]!='.':
    #     final_text = final_text + ' <mask>'

    return final_text.strip() # 去掉连续的mask，得到一个最终text


def mask_entities(text, labels, shouldMask):
    '''
        Mask some part of the continuous entities 对实体类型进行BI标签输入
    '''

    start_index = -1

    for i in range(len(text)):
        if labels[i] == 'O' and start_index!= -1: 
            mask_indices = find_masks_for_entities(start_index, i-1) if shouldMask==True else []
            for j in range(start_index, i):
                if j in mask_indices and shouldMask:
                    text[j] = '<' + labels[j].lower() + '> <mask> <' + labels[j].lower() + '>'
                else:
                    text[j] = '<' + labels[j].lower() + '> ' + text[j] + ' <' + labels[j].lower() + '>' # 在text列中为实体打上标签
            start_index = -1
        elif labels[i] != 'O' and start_index==-1: # 找到labels中B开头
            start_index = i
    # 进行第三步：Labelled Sequence Linearization
    if start_index!=-1:
        mask_indices = find_masks_for_entities(start_index, len(text)-1) if shouldMask==True else []
        for j in range(start_index, len(text)):
            if j in mask_indices and shouldMask:
                text[j] = '<' + labels[j].lower() + '> <mask> <' + labels[j].lower() + '>'
            else:
                text[j] = '<' + labels[j].lower() + '> ' + text[j] + ' <' + labels[j].lower() + '>'
    return text

def mask_words(text, labels, attn, attnMode='none'):
    '''
        Mask parts of continuous attn words and useless words 对非关键字和实体类型的词进行mask操作
    '''
    # print(len(text),len(labels), len(attn))
    if attnMode == 'none':
        for i in range(len(text)):
            if labels[i]=='O':
                text[i] = '<mask>'
    elif attnMode == 'all':
        for i in range(len(text)):
            if attn[i]=='0':
                text[i] = '<mask>'
    elif attnMode == 'gauss':
        mask_attn_indices = []
        mask_useless_words = []
        for i in range(len(text)):
            if attn[i]=='1' and labels[i]=='O': # 关键字的判断
                mask_attn_indices.append(i)
            elif labels[i]=='O':                # 无用词的判断
                mask_useless_words.append(i)    

        mask_count = 0
        if len(mask_attn_indices):
            mask_rate = get_random_gauss_value(0.5, 1/len(mask_attn_indices)) # 高斯分布中采样动态掩码率
            mask_count = round(len(mask_attn_indices)*mask_rate) #根据mask_rate 从句子中的K个关键字中随机抽取mask_count数量token

        mask_indices = random.sample(mask_attn_indices, mask_count) + mask_useless_words #  random.sample 函数从 mask_attn_indices 列表中随机选择 mask_count 个元素，并将它们添加到 mask_indices 中 + mask_useless_words

        for i in mask_indices:
            text[i] = '<mask>'

    return merge_list(text) # 返回的是一个已经打好mask和打好实体标签的text


sentence = "joplin 's recent scuffle with john stillwell stark over the publication of the ragtime dance created a level of animosity between composer and publisher ."
labels = 'O O O O O B-PER I-PER I-PER O O O O B-CW I-CW I-CW O O O O O O O O O O'
attn = '1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 1 0 1'
