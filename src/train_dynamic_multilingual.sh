#!/bin/bash

# All languages and their keys in m-bart

# Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE),
# Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT),
# Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM),
#  Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK),
#  Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ),
#  Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID),
#  Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN),
#  Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE),
#  Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA),
#  Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)
# sh train_dynamic_multilingual.sh zh zh_CN 100 8 42 0.3 5
set -e
set -x

language=$1             zh
language_label=$2       zh_CN
size=$3                 100
flair_batch_size=$4     8
SEED=$5                 42
masking_rate=$6         0.3
generations=$7          5

input_folder="../data/${size}"                          "../data/100"

python flair_train.py \
-i $input_folder \                                      "../data/100"
-o "${input_folder}/${language}_flair_xlm_${size}" \    "../data/100/zh_flair_xlm_100"
-g cuda:0 \
-tf "${language}_sample_train.conll" \                  "zh_sample_train.conll"
-bs $flair_batch_size \                                  8
-l 0.01 \
-ep 100 \
-lang $language \                                       'zh'
-s $SEED                                                 42

python generate-bert-attn.py \
-a attn \
-m $masking_rate \                                                      0.3
-dir "../data/${size}" \                                               "../data/100"
-ckpt "../data/${size}/${language}_flair_xlm_${size}/best-model.pt" \   "../data/100/zh_flair_xlm_100/best-model.pt"
-tf "../data/${size}/${language}_sample_train.conll" \                  "../data/100/zh_sample_train.conll"
-df "../data/${size}/${language}_dev.conll"                             "../data/100/zh_dev.conll"

directory="../data/${size}"                                             "../data/100"
attn_train="${language}_sample_train_attn_${masking_rate}_xlm-roberta-large"  "zh_sample_train_attn_0.3_xlm-roberta-large"
attn_dev="${language}_dev_attn_${masking_rate}_xlm-roberta-large"       "zh_dev_attn_0.3_xlm-roberta-large"

run="${masking_rate}-false-gauss-attention-dynamic-${masking_rate}-${generations}-false-${size}-xlm-large-${language}-${SEED}-retrain"
                                                                 #    "0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"
python bart_pretrain_dynamic_multilingual.py \
--directory $directory \          "../data/100"
--train_file $attn_train \        "zh_sample_train_attn_0.3_xlm-roberta-large"
--dev_file $attn_dev \            "zh_dev_attn_0.3_xlm-roberta-large"
--epochs 10 \                     10
--batch_size 16 \                 16
--mask_entities False \           False
--mask_attn gauss \               gauss
--mode attn \                     att
--file_name $run \                "0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"
--lang $language_label \          "zh_CN"
--seed $SEED                      42

best_model="${directory}/${attn_train}-${run}-final"     "../data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-final"

inference_file="${language}_sample_train_attn_${masking_rate}_xlm-roberta-large"   "zh_sample_train_attn_0.3_xlm-roberta-large"

python test-dynamic_multilingual.py \
--model $best_model \                    "../data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-final"
--input_file $inference_file \            "zh_sample_train_attn_0.3_xlm-roberta-large"
--sample_generation_mode dynamic \        "dynamic"
--directory $directory \                  "../data/100"
--mask_entities False \
--mask_attn gauss \
--mode attn \
--topk 10 \
--num_of_sequences $generations \          5
--max_length 100 \
--do_sample True \
--num_beams 5 \
--file_name $run \                          "0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"
--root_dir $directory \                    "../data/100"
--lang $language_label \                   "zh_CN"
--remove_repetitions False \
--seed $SEED                                42

generated_file="${inference_file}-${run}"    "zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"

python flair_eval_equal.py \
--input_folder $directory \                                               "../data/100"
--output_folder "${directory}/${generated_file}" \                        "../data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"
--gpu cuda:0 \
--input_file $generated_file \                                           "zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"
--need_consistency True \
--file_name $run \                                                       "0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain"
-gfl $language \                                                         "zh"
--seed $SEED \                                                           42
--ckpt "../data/${size}/${language}_flair_xlm_${size}/best-model.pt"     "../data/100/zh_flair_xlm_100/best-model.pt"

consistent_file="${generated_file}-aug+gold.txt"                         "zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-aug+gold.txt"

python flair_train.py \
--input_folder $directory \                                              "../data/100"
--output_folder "${directory}/${generated_file}-flair" \                 "../data/100/zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-flair"
--gpu cuda:0 \
--train_file $consistent_file \                                           "zh_sample_train_attn_0.3_xlm-roberta-large-0.3-false-gauss-attention-dynamic-0.3-5-false-100-xlm-large-zh-42-retrain-aug+gold.txt"
--batch_size $flair_batch_size \                                          8
--lr 0.01 \
--epochs 100 \
--language $language \                                                    "zh"
--seed $SEED                                                              42
