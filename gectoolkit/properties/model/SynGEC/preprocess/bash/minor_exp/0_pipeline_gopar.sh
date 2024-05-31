#!/bin/bash

# 更换数据集
dataset="falko"
export CUDA_VISIBLE_DEVICES=6

current_directory=$(pwd)
echo "当前目录是：$current_directory"

path="../../../../gectoolkit/properties/model/Syngec"
data_dir="${path}/data/${dataset}"

# 训练集src,tgt,m2
train_src_file=$data_dir/trainset/src.txt
train_tgt_file=$data_dir/trainset/tgt.txt
train_m2_file=$data_dir/trainset/m2_reversed.txt

# 验证集src,tgt,m2
valid_src_file=$data_dir/validset/src.txt
valid_tgt_file=$data_dir/validset/tgt.txt
valid_m2_file=$data_dir/validset/m2_reversed.txt

# 测试集src,tgt,m2
test_src_file=$data_dir/testset/src.txt
test_tgt_file=$data_dir/testset/tgt.txt
test_m2_file=$data_dir/testset/m2_reversed.txt

train_path="${path}/data/$dataset/trainset/src.txt.conll_converted_gopar"
valid_path="${path}/data/$dataset/validset/src.txt.conll_converted_gopar"
test_path="${path}/data/$dataset/testset/src.txt.conll_converted_gopar"

###########################################################################
# You need python==3.10,transformers-4.38.2
# pip from source "pip install -U git+https://github.com/yzhangcs/parser"
# download: parser = Parser.load('dep-biaffine-xlmr',reload=True)
###########################################################################
vanilla_parser_path=dep-biaffine-xlmr
gopar_path="${path}/model/gopar/biaffine-dep-electra-xlmr-gopar-${dataset}-test"
command_path="../../../../gectoolkit/model/SynGEC/src/src_gopar"

############################################################################################
# Step 1. Parse the target-side sentences in parallel GEC data by an off-the-shelf parser
# environment: supar2
# If you find this step cost too much time,you can split the large file to several small files and predict them on multiple GPUs,and then merge the results.
############################################################################################
echo "------ Start to pipeline ${dataset} gopar ---------"
echo "step1: Parse target-side sentences..."
python ${command_path}/parse.py $train_tgt_file $train_tgt_file.conll_predict $vanilla_parser_path&& \
    echo "step 1: Parse target-side sentences $train_tgt_file.conll_predict successful!"
python ${command_path}/parse.py $valid_tgt_file $valid_tgt_file.conll_predict $vanilla_parser_path&& \
    echo "step 1: Parse target-side sentences $valid_tgt_file.conll_predict successful!"
python ${command_path}/parse.py $test_tgt_file $test_tgt_file.conll_predict $vanilla_parser_path&& \
    echo "step 1: Parse target-side sentences $test_tgt_file.conll_predict successful!"

####################################################################
# Step 2. Extract edits by ERRANT from target-side to source-side
# environment: supar2
# If you meet this error: `OSError: [E050] Can't find model 'en'.`
# Please first run this command: `python -m spacy download en`
#####################################################################
echo "Step 2: Extract edits by ERRANT..."
errant_parallel -orig $train_tgt_file -cor $train_src_file -out $train_m2_file&& \
    echo "step 2: Extract edits by ERRANT $train_m2_file successful!"
errant_parallel -orig $valid_tgt_file -cor $valid_src_file -out $valid_m2_file&& \
    echo "step 2: Extract edits by ERRANT $valid_m2_file successful!"
errant_parallel -orig $test_tgt_file -cor $test_src_file -out $test_m2_file&& \
    echo "step 2: Extract edits by ERRANT $test_m2_file successful!"

####################################################################
# Step 3. Project the target-side trees to source-side ones
# environment: supar2
####################################################################
echo "Step 3: Project the target-side trees..."
python ${command_path}/convert_gec_data_to_parsing_data_english.py $train_tgt_file.conll_predict $train_m2_file $train_src_file.conll_converted_gopar&& \
    echo "step 3: Project $train_src_file.conll_converted_gopar successful!"
python ${command_path}/convert_gec_data_to_parsing_data_english.py $valid_tgt_file.conll_predict $valid_m2_file $valid_src_file.conll_converted_gopar&& \
    echo "step 3: Project $valid_src_file.conll_converted_gopar successful!"
python ${command_path}/convert_gec_data_to_parsing_data_english.py $test_tgt_file.conll_predict $test_m2_file $test_src_file.conll_converted_gopar&& \
    echo "step 3: Project $valid_src_file.conll_converted_gopar successful!"

############################################################################################
## Step 4. Train GOPar
#   syngec的环境下，连接vpn！！！！
############################################################################################
mkdir -p $gopar_path
python -m supar.cmds.biaffine_dep train -b -d ${CUDA_VISIBLE_DEVICES} \
       -c ../../src/src_gopar/configs/ptb.biaffine.dep.electra.ini \
       -p $gopar_path/model \
       -f char \
       --encoder bert \
       --bert google/electra-large-discriminator \
       --train ${train_path} \
       --dev ${dev_path} \
       --test ${test_path} \
       --seed 1 \
       --punct
