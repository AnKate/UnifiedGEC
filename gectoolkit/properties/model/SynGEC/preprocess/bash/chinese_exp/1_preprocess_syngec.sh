#################################
# Preprocess train valid
# 需要修改 $FAIRSEQ_DIR/preprocess.py文件中的 bpe换成char
###############################
CUDA_DEVICE=3
dataset=nlpcc18_translation

FAIRSEQ_DIR="../../../../gectoolkit/model/SynGEC/src/src_syngec/fairseq2/fairseq_cli"


## # todo: 改dataset 训练集的proprecess
PROCESSED_DIR=../../preprocess_data/chinese_${dataset}_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

## todo: 改dataset File path
TRAIN_SRC_FILE=../../data/${dataset}/trainset/src.txt
TRAIN_TGT_FILE=../../data/${dataset}/trainset/tgt.txt
VALID_SRC_FILE=../../data/${dataset}/validset/src.txt
VALID_TGT_FILE=../../data/${dataset}/validset/tgt.txt

wc -l ../../data/${dataset}/trainset/src.txt
wc -l ../../data/${dataset}/trainset/tgt.txt

###################################################################
### step1: apply char
## 输入：      ../../data/nlpcc18/trainset/src.txt
## 分词后的结果 ../../data/nlpcc18/trainset/src.txt.char
##################################################################
echo "step1: apply char..."
python ../../utils/segment_bert.py <$TRAIN_SRC_FILE >$TRAIN_SRC_FILE".char"&&\
    echo "step1: apply char $TRAIN_SRC_FILE".char" successful"
python ../../utils/segment_bert.py <$TRAIN_TGT_FILE >$TRAIN_TGT_FILE".char"&&\
    echo "step1: apply char $TRAIN_TGT_FILE".char" successful"
python ../../utils/segment_bert.py <$VALID_SRC_FILE >$VALID_SRC_FILE".char"&&\
    echo "step1: apply char $VALID_SRC_FILE".char" successful"
python ../../utils/segment_bert.py <$VALID_TGT_FILE >$VALID_TGT_FILE".char"&&\
    echo "step1: apply char $VALID_TGT_FILE".char" successful"

###################################################################
### step2: Subword Align
#  输入:   ../../data/nlpcc18/trainset/src.txt.char
# 对齐后:  ../../data/nlpcc18/trainset/src.txt.swm
###################################################################
echo "step2: Align subwords and words..."
python ../../utils/subword_align.py $TRAIN_SRC_FILE".char" $TRAIN_SRC_FILE".char" $TRAIN_SRC_FILE".swm"&&\
    echo "step2: Align to $TRAIN_SRC_FILE".swm" successful"
python ../../utils/subword_align.py $VALID_SRC_FILE".char" $VALID_SRC_FILE".char" $VALID_SRC_FILE".swm"&&\
    echo "step2: Align to $VALID_SRC_FILE".swm" successful"

###################################################################
#### fairseq preprocess
###################################################################
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".char" $PROCESSED_DIR/train.char.src

cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".char" $PROCESSED_DIR/train.char.tgt

cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".char" $PROCESSED_DIR/valid.char.src

cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".char" $PROCESSED_DIR/valid.char.tgt

### 输入： ../../data/nlpcc18/trainset/src.txt.swm
### 输出： ../../preprocess/chinese_nlpcc18_with_syntax_transformer/train.swm.src
cp $TRAIN_SRC_FILE".swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".swm" $PROCESSED_DIR/valid.swm.src

###################################################################
## step3: syntax specific
## 输入： ../../data/nlpcc18/trainset/src.txt.char.conll_predict_gopar
## 输出： ../../data/nlpcc18/trainset/src.txt.char.conll_predict_gopar_np
###################################################################
echo "step3: syntax specific..."
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE "char.${CoNLL_SUFFIX}" conll transformer
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE "char.${CoNLL_SUFFIX}" probs transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE "char.${CoNLL_SUFFIX}" conll transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE "char.${CoNLL_SUFFIX}" probs transformer


## 复制的时候
##       输入： ../../data/nlpcc18/trainset/src.txt.char.conll_predict_gopar_np
## 复制到的路径：../../preprocess/chinese_nlpcc18_with_syntax_transformer/train.conll.src
cp $TRAIN_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
cp $VALID_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src
echo "step3: syntax specific $PROCESSED_DIR/train.conll.src successful"
echo "step3: syntax specific $PROCESSED_DIR/valid.conll.src successful"


############################################################################
# step4: Calculate dependency distance..
# 输入： ../../data/nlpcc18/trainset/src.txt.char.conll_predict_gopar
# 输出： $TRAIN_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd
#############################################################################

echo "step4: Calculate dependency distance..."
python ../../utils/calculate_dependency_distance.py $TRAIN_SRC_FILE".char.${CoNLL_SUFFIX}" $PROCESSED_DIR/train.swm.src $TRAIN_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd"&&\
    echo "step4: Calculate dependency to $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" successful"
python ../../utils/calculate_dependency_distance.py $VALID_SRC_FILE".char.${CoNLL_SUFFIX}" $PROCESSED_DIR/valid.swm.src $VALID_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd"&&\
    echo "step4: Calculate dependency to  $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" successful"


cp $TRAIN_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/train.dpd.src
cp $VALID_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/valid.dpd.src

cp $TRAIN_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/train.probs.src
cp $VALID_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/valid.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../../../gectoolkit/model/SynGEC/src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --trainpref $PROCESSED_DIR/train.char \
       --validpref $PROCESSED_DIR/valid.char \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../dicts/syntax_label_gec.dict \
       --srcdict ../../dicts/chinese_vocab.count.txt \
       --tgtdict ../../dicts/chinese_vocab.count.txt&&\
       echo " -------- $PROCESSED_DIR/bin successful!!"


##########################
## Preprocess nlpcc18 test
##########################

FAIRSEQ_DIR="../../../../gectoolkit/model/SynGEC/src/src_syngec/fairseq2/fairseq_cli"
## 测试集的proprecess
PROCESSED_DIR=../../preprocess_data/chinese_${dataset}_test_with_syntax_bart

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

### File path
TEST_SRC_FILE=../../data/${dataset}/testset/src.txt
TEST_TGT_FILE=../../data/${dataset}/testset/tgt.txt


### Subword Align
echo "step2: Align subwords and words..."
python ../../utils/subword_align.py $TEST_SRC_FILE".char" $TEST_SRC_FILE".char" $TEST_SRC_FILE".swm"&&\
    echo "step2: Align to $TEST_SRC_FILE".swm" successful"


#### fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".char" $PROCESSED_DIR/test.char.src
cp $TEST_SRC_FILE".swm" $PROCESSED_DIR/test.swm.src
cp $TEST_TGT_FILE".char" $PROCESSED_DIR/test.char.tgt


### syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE "char.$CoNLL_SUFFIX" conll transformer
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE "char.$CoNLL_SUFFIX" probs transformer

cp $TEST_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/test.conll.src
echo "step3: syntax specific $PROCESSED_DIR/test.conll.src successful"


# Calculate dependency distance
echo "step4: Calculate dependency distance..."
python ../../utils/calculate_dependency_distance.py $TEST_SRC_FILE".char.${CoNLL_SUFFIX}" $PROCESSED_DIR/test.swm.src $TEST_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd"&&\
    echo "step4: Calculate dependency to $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" successful"


cp $TEST_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/test.dpd.src
cp $TEST_SRC_FILE".char.${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/test.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../../../gectoolkit/model/SynGEC/src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --only-source \
       --testpref $PROCESSED_DIR/test.char \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../dicts/syntax_label_gec.dict \
       --srcdict ../../dicts/chinese_vocab.count.txt \
       --tgtdict ../../dicts/chinese_vocab.count.txt&&\
       echo " -------- $PROCESSED_DIR/bin successful!!"