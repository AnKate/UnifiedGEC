#!/bin/bash

# todo:edit dataset and language
dataset='falko'
language='german'

export CUDA_VISIBLE_DEVICES=2

path="../../../../gectoolkit/properties/model/Syngec"
command_path="../../../../gectoolkit/model/SynGEC/src"
gopar_path="${path}/model/gopar/biaffine-dep-electra-xlmr-gopar-${dataset}-test"

FAIRSEQ_DIR="../../../../gectoolkit/model/SynGEC/src/src_syngec/fairseq2/fairseq_cli"
PROCESSED_DIR="${path}/model/syngec/${language}_${dataset}_with_syntax_bart"

WORKER_NUM=32
DICT_SIZE=32000

CoNLL_SUFFIX="conll_predict_gopar"
CoNLL_SUFFIX_PROCESSED="conll_predict_gopar_bart_np"

TRAIN_SRC_FILE="${path}/data/${dataset}/trainset/src.txt"
TRAIN_TGT_FILE="${path}/data/${dataset}/trainset/tgt.txt"
VALID_SRC_FILE="${path}/data/${dataset}/validset/src.txt"
VALID_TGT_FILE="${path}/data/${dataset}/validset/tgt.txt"


###############################################################
### Step 5. Predict source-side trees for GEC training
### need to use environment(supar2)
################################################################
#echo "Step 5. Predict source-side trees......"
#
# {dataset} trainset (training)
IN_FILE=${path}/data/$dataset/trainset/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"

## {dataset} validset (valid)
IN_FILE=${path}/data/$dataset/validset/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"
#
## {dataset} testset (test)
IN_FILE=${path}/data/$dataset/testset/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
python ${command_path}/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"


##############################################################
## Step 6. apply bpe
## need to use environment(syngec)
###############################################################
if [ ! -f $TRAIN_SRC_FILE".bart_bpe" ]; then
    echo "step6:  Apply BPE..."
    python ${command_path}/src_syngec/fairseq2/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TRAIN_SRC_FILE \
        --outputs $TRAIN_SRC_FILE".bart_bpe" \
        --workers 60 \
        --keep-empty &&\
    echo "step6: BPE applied to $TRAIN_SRC_FILE".bart_bpe" successful"
    python ${command_path}/src_syngec/fairseq2/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TRAIN_TGT_FILE \
        --outputs $TRAIN_TGT_FILE".bart_bpe" \
        --workers 60 \
        --keep-empty &&\
    echo "step6: BPE applied to $TRAIN_TGT_FILE".bart_bpe" successful"
    python ${command_path}/src_syngec/fairseq2/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $VALID_SRC_FILE \
        --outputs $VALID_SRC_FILE".bart_bpe" \
        --workers 60 \
        --keep-empty &&\
    echo "step6: BPE applied to $VALID_SRC_FILE".bart_bpe" successful"
    python ${command_path}/src_syngec/fairseq2/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $VALID_TGT_FILE \
        --outputs $VALID_TGT_FILE".bart_bpe" \
        --workers 60 \
        --keep-empty &&\
    echo "step6: BPE applied to $VALID_TGT_FILE".bart_bpe" successful"
fi

####################################################
## step7: Decode
####################################################
if [ ! -f $TRAIN_SRC_FILE".bart_bpe.tok" ]; then
    echo "step7:  Decode BPE..."
    python ../../utils/multiprocessing_bpe_decoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TRAIN_SRC_FILE".bart_bpe" \
        --outputs $TRAIN_SRC_FILE".bart_bpe.tok" \
        --workers 60 \
        --keep-empty &&\
    echo "step7: BPE decode to $TRAIN_SRC_FILE".bart_bpe.tok" successful"
    python ../../utils/multiprocessing_bpe_decoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $VALID_SRC_FILE".bart_bpe" \
        --outputs $VALID_SRC_FILE".bart_bpe.tok" \
        --workers 60 \
        --keep-empty &&\
    echo "step7: BPE decode to $VALID_SRC_FILE".bart_bpe.tok" successful"
fi


###############################
## step8:Subword Align
####################################
if [ ! -f $TRAIN_SRC_FILE".bart_swm" ]; then
    echo "step8: Align subwords and words..."
    python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bart_bpe.tok" $TRAIN_SRC_FILE".bart_swm" &&\
    echo "step8: Align to $TRAIN_SRC_FILE".bart_swm" successful"
    python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bart_bpe.tok" $VALID_SRC_FILE".bart_swm" &&\
    echo "step8: Align to $TRAIN_SRC_FILE".bart_swm" successful"
fi


mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.tgt

cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".bart_swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".bart_swm" $PROCESSED_DIR/valid.swm.src

########################################
### step9: syntax specific
########################################

python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX probs bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX probs bart

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
echo "step9: syntax specific $PROCESSED_DIR/train.conll.src successful"

cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src
echo "step9: syntax specific $PROCESSED_DIR/valid.conll.src successful"


###########################################
## step 10:Calculate dependency distance
##############################################
if [ ! -f $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
    echo "step10: Calculate dependency distance..."
    python ../../utils/calculate_dependency_distance.py $TRAIN_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/train.swm.src $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"&&\
    echo "step10: Calculate dependency to $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" successful"
    python ../../utils/calculate_dependency_distance.py $VALID_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/valid.swm.src $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"&&\
    echo "step10: Calculate dependency to  $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" successful"
fi

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/train.dpd.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/valid.dpd.src

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/train.probs.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/valid.probs.src


### if lanuage=='chinese',# 376 train_src_file = args.trainpref.replace("char", suffix) + ".src"  # 11.10改
### else train_src_file = args.trainpref.replace("bpe", suffix) + ".src"  #作者的
echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin
python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
    --user-dir ${command_path}/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --trainpref $PROCESSED_DIR/train.bpe \
    --validpref $PROCESSED_DIR/valid.bpe \
    --destdir $PROCESSED_DIR/bin \
    --workers $WORKER_NUM \
    --conll-suffix conll \
    --swm-suffix swm \
    --dpd-suffix dpd \
    --probs-suffix probs \
    --labeldict ../../dicts/syntax_label_gec.dict \
    --srcdict ../../pretrained_weights/dict.txt \
    --tgtdict ../../pretrained_weights/dict.txt &&\
    echo "----- $dataset Train valid Finished! successful -----"


###########################################################################
## Preprocess test
###########################################################################

TEST_SRC_FILE="${path}/data/${dataset}/testset/src.txt"
TEST_TGT_FILE="${path}/data/${dataset}/testset/tgt.txt"

PROCESSED_DIR="${path}/model/syngec/${language}_${dataset}_test_with_syntax_bart"

echo "Preprocess test..."
# apply bpe
if [ ! -f $TEST_SRC_FILE".bart_bpe" ]; then
    echo "step1:  Apply BPE..."
    python ${command_path}/src_syngec/fairseq2/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TEST_SRC_FILE \
        --outputs $TEST_SRC_FILE".bart_bpe" \
        --workers 60 \
        --keep-empty &&\
        echo "step1: BPE applied to $TEST_SRC_FILE".bart_bpe" successful"
    python ${command_path}/src_syngec/fairseq2/examples/roberta/multiprocessing_bpe_encoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TEST_TGT_FILE \
        --outputs $TEST_TGT_FILE".bart_bpe" \
        --workers 60 \
        --keep-empty &&\
        echo "step1: BPE applied to $TEST_TGT_FILE".bart_bpe" successful"
fi

# Decode
if [ ! -f $TEST_SRC_FILE".bart_bpe.tok" ]; then
    echo "step2:  Decode BPE..."
    python ../../utils/multiprocessing_bpe_decoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TEST_SRC_FILE".bart_bpe" \
        --outputs $TEST_SRC_FILE".bart_bpe.tok" \
        --workers 60 \
        --keep-empty &&\
        echo "step2: BPE decode to $TEST_SRC_FILE".bart_bpe.tok" successful"
    python ../../utils/multiprocessing_bpe_decoder.py \
        --encoder-json ../../pretrained_weights/encoder.json \
        --vocab-bpe ../../pretrained_weights/vocab.bpe \
        --inputs $TEST_TGT_FILE".bart_bpe" \
        --outputs $TEST_TGT_FILE".bart_bpe.tok" \
        --workers 60 \
        --keep-empty &&\
        echo "step2: BPE decode to $TEST_TGT_FILE".bart_bpe.tok" successful"
fi

# step3:Subword Align
if [ ! -f $TEST_SRC_FILE".bart_swm" ]; then
    echo "step3: Align subwords and words..."
    python ../../utils/subword_align.py $TEST_SRC_FILE $TEST_SRC_FILE".bart_bpe.tok" $TEST_SRC_FILE".bart_swm" &&\
    echo "step3: Align to $TEST_SRC_FILE".bart_swm" successful"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".bart_bpe" $PROCESSED_DIR/test.bpe.src
cp $TEST_SRC_FILE".bart_swm" $PROCESSED_DIR/test.swm.src

cp $TEST_TGT_FILE $PROCESSED_DIR/test.tgt
cp $TEST_TGT_FILE".bart_bpe" $PROCESSED_DIR/test.bpe.tgt


# step5:syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX probs bart

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/test.conll.src

# step6: Calculate dependency distance
if [ ! -f $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
    echo "step6: Calculate dependency distance..."
    python ../../utils/calculate_dependency_distance.py $TEST_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/test.swm.src $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" &&\
    echo "step6: Calculate dependency to  $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd successful
fi

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/test.dpd.src
cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/test.probs.src


echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

#    --only-source \
python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
    --user-dir ${command_path}/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --testpref $PROCESSED_DIR/test.bpe \
    --destdir $PROCESSED_DIR/bin \
    --workers $WORKER_NUM \
    --conll-suffix conll \
    --swm-suffix swm \
    --dpd-suffix dpd \
    --probs-suffix probs \
    --labeldict ../../dicts/syntax_label_gec.dict \
    --srcdict ../../pretrained_weights/dict.txt \
    --tgtdict ../../pretrained_weights/dict.txt &&\
    echo "----- ${dataset} Finished——test! successful -----"