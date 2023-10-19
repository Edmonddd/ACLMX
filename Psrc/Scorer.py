import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM, XLMRobertaTokenizer, XLMRobertaForMaskedLM
import json
import copy
import gc
from tqdm import tqdm

model_path = "/home/hsq/ACLM-main/xlm-roberta-large"
max_len = 120
device = 'cuda'
with torch.no_grad():
    model = XLMRobertaForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)


def bert_ppl_original(sent):
    tokenize_input = tokenizer.tokenize(sent)
    sen_len = len(tokenize_input)

    if sen_len < max_len:
        tensor_input = tokenizer.convert_tokens_to_ids(tokenize_input)
        all_input = []
        for i, word in enumerate(tensor_input):
            text = copy.copy(tensor_input)
            text[i] = tokenizer.unk_token_id
            all_input.append(text)
        all_input = torch.tensor(all_input).to(device)
        output = model(all_input)
        pred_scores = output[0]
        # print(pred_scores.shape)
        index1 = torch.tensor([[_] for _ in range(pred_scores.shape[0])])
        index2 = torch.tensor([[_] for _ in tensor_input])
        probs = pred_scores[index1, index1].squeeze(1)
        probs = torch.log_softmax(probs, dim=1)
        word_loss = probs[index1, index2]
        sent_loss = torch.sum(word_loss).item()
        ppl = np.exp(-sent_loss / sen_len)
        del tokenize_input, tensor_input, all_input, output, pred_scores, index1, index2, probs, word_loss, sent_loss
        gc.collect()
        return ppl

    else:
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sent_loss = 0.
        for i, word in enumerate(tokenize_input):
            tokenize_input[i] = tokenizer.mask_token
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            output = model(mask_input)
            pred_scores = output[0]
            ps = torch.log_softmax(pred_scores[0, i], dim=0)
            word_loss = ps[tensor_input[0, i]]
            sent_loss += word_loss.item()
            tokenize_input[i] = word  # restore
        ppl = np.exp(-sent_loss / sen_len)
        return ppl


def create_score(filepath):
    setence_socre_ls = []
    with open(filepath, 'w') as f:
        for sentence in tqdm(f):
            score = bert_ppl_original(sentence)
            tmp = {sentence: score}
            setence_socre_ls.append(tmp)
            f.write(score)


create_score()


# sentence = 'Xinhua News Agency , Canberra , January 13 , by Xinhua News Agency , in Manila , by reporter Changyi Xiong'
# score = bert_ppl_original(sentence)
# print(score)