#data_dir=../../data/clang8_train
#data_dir=../../data/wi_locness_train
#src_file=$data_dir/src.txt.original
#tgt_file=$data_dir/tgt.txt
#m2_file=$data_dir/m2_reversed.txt
#vanilla_parser_path=biaffine-dep-roberta-en
#vanilla_parser_path=/mnt/nas_alinlp/zuyi.bzy/zhangyue/syntactic_GEC/tools/parser/biaffine-dep-roberta-en/biaffine-dep-roberta-en-gec
#gopar_path=../../model/gopar/biaffine-dep-electra-en-gopar-test
#
## Step 1. Parse the target-side sentences in parallel GEC data by an off-the-shelf parser
### If you find this step cost too much time, you can split the large file to several small files and predict them on multiple GPUs, and then merge the results.
#python ../../src/src_gopar/parse.py $tgt_file $tgt_file.conll_predict $vanilla_parser_path
#
## Step 2. Extract edits by ERRANT from target-side to source-side
### If you meet this error: `OSError: [E050] Can't find model 'en'.`
### Please first run this command: `python -m spacy download en`
#errant_parallel -orig $tgt_file -cor $src_file -out $m2_file
#
## Step 3. Project the target-side trees to source-side ones
#python ../../src/src_gopar/convert_gec_data_to_parsing_data_english.py $tgt_file.conll_predict $m2_file $src_file.conll_converted_gopar
#
## Step 4. Train GOPar
### You should also re-run the 1-3 steps to generate dev/test data (BEA19-dev & CoNLL14-test)

#mkdir -p $gopar_path
#python -m torch.distributed.launch --nproc_per_node=8 --master_port=10000 \
#       -m supar.cmds.biaffine_dep train -b -d 0,1,2,3,4,5,6,7 \
#       -c ../../src/src_gopar/configs/ptb.biaffine.dep.electra.ini \
#       -p $gopar_path/model \
#       -f char \
#       --encoder bert \
#       --bert google/electra-large-discriminator \
#       --train $src_file.conll_converted_gopar \
#       --dev ../../data/bea19_dev/src.txt.original.conll_converted_gopar \
#       --test ../../data/conll14_test/src.txt.original.conll_converted_gopar \
#       --seed 1 \
#       --punct


#train_path="../../data/$dataset/trainset/src.txt.conll_converted_gopar"
#dev_path="../../data/$dataset/validset/src.txt.conll_converted_gopar"
#test_path="../../data/$dataset/testset/src.txt.conll_converted_gopar"
#
#mkdir -p $gopar_path
#python -m supar.cmds.biaffine_dep train -b -d 3 \
#       -c ../../src/src_gopar/configs/ptb.biaffine.dep.electra.ini \
#       -p $gopar_path/model \
#       -f char \
#       --encoder bert \
#       --bert google/electra-large-discriminator \
#       --train $train_path \
#       --dev $dev_path \
#       --test $test_path \
#       --seed 1 \
#       --punct


###############################################################
# Step 5. Predict source-side trees for GEC training(vpn)
###############################################################

dataset='conll14_translation'
language='english'

export CUDA_VISIBLE_DEVICES=5

command_path="../../../../gectoolkit/model/SynGEC/src"
FAIRSEQ_DIR="../../../../gectoolkit/model/SynGEC/src/src_syngec/fairseq2/fairseq_cli"
gopar_path="../../model/gopar/biaffine-dep-electra-en-gopar"

WORKER_NUM=32
DICT_SIZE=32000

CoNLL_SUFFIX="conll_predict_gopar"
CoNLL_SUFFIX_PROCESSED="conll_predict_gopar_bart_np"

echo "Step 5. Predict source-side trees......"

## {dataset} trainset (training)
IN_FILE=../../data/$dataset/trainset/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"

## {dataset} validset (valid)
IN_FILE=../../data/$dataset/validset/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"
#
## {dataset} testset (test)
IN_FILE=../../data/$dataset/testset/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"



#dataset='conll14_augment_xinyuan'
#language='english'
#echo "Step 5. Predict ${dataset} source-side trees......"
#
### {dataset} trainset (training)
#IN_FILE=../../data/$dataset/trainset/src.txt
#OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
#python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path &&\
#    echo "Step 5. Predict source-side trees $OUT_FILE successful!"
#
### {dataset} validset (valid)
#IN_FILE=../../data/$dataset/validset/src.txt
#OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
#python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path &&\
#    echo "Step 5. Predict source-side trees $OUT_FILE successful!"
##
### {dataset} testset (test)
#IN_FILE=../../data/$dataset/testset/src.txt
#OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
#python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path &&\
#    echo "Step 5. Predict source-side trees $OUT_FILE successful!"
