#dataset=nlpcc18_part
#data_dir=../../data/${dataset}/validset
#src_file=$data_dir/src.txt             # 错误句子文本     ../../data/nlpcc18/trainset/src.txt.original
#tgt_file=$data_dir/tgt.txt             # 正确句子文本     ../../data/nlpcc18/trainset/tgt.txt
#src_file_char=$data_dir/src.txt.char   # 分词后的错误句子  ../../data/nlpcc18/trainset/src.txt.original.char
#tgt_file_char=$data_dir/tgt.txt.char   # 分词后的正确句子  ../../data/nlpcc18/trainset/tgt.txt.char
#
#para_file=$data_dir/para_tgt2src.txt   # ../../data/nlpcc18/trainset/para_tgt2src.txt
#m2_file=$data_dir/m2_reversed.txt      # ../../data/nlpcc18/trainset/m2_reversed.txt
#
## 现有解析器的路径
##vanilla_parser_path=../../model/gopar/biaffine-dep-electra-zh-char
#
## gopar解析器的路径
#gopar_path=../../model/gopar/biaffine-dep-electra-zh-gopar


# Step 1. Parse the target-side sentences in parallel GEC data by an off-the-shelf parser
# If you find this step cost too much time, you can split the large file to several small files and predict them on multiple GPUs, and merge the results.
# 输入：../../data/nlpcc18/trainset/tgt.txt.char
# 输出：../../data/nlpcc18/trainset/tgt.txt.char.conll_predict
#python ../../src/src_gopar/parse.py $tgt_file_char $tgt_file.conll_predict $vanilla_parser_path
#
#
## Step 2. Extract edits by ChERRANT from target-side to source-side
#cherrant_path=./scorers/ChERRANT  # You need to first download ChERRANT from https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT
#
#cherrant_path=../../scorers/ChERRANT
#cd $cherrant_path
#paste $tgt_file_char $src_file_char | awk '{print NR"\t"$p}' > $para_file
#python parallel_to_m2.py -f $para_file -o $m2_file -g char --segmented
#cd -
##
#
###Step 3. Project the target-side trees to source-side ones
#python ../../src/src_gopar/convert_gec_data_to_parsing_data_chinese.py $tgt_file.conll_predict $m2_file $src_file.conll_converted_gopar
#
## Step 4. Train GOPar
#mkdir -p $gopar_path
#python -m torch.distributed.launch --nproc_per_node=8 --master_port=10000 \
#       -m supar.cmds.biaffine_dep train -b -d 0,1,2,3,4,5,6,7 -c ../../src/src_gopar/configs/ctb7.biaffine.dep.electra.ini -p $gopar_path/model -f char --encoder bert --bert hfl/chinese-electra-180g-large-discriminator \
#       --train $src_file.conll_converted_gopar \
#       --dev ../../data/mucgec_dev/src.txt.conll_converted_gopar \
#       --test ../../data/mucgec_dev/src.txt.conll_converted_gopar \
#       --seed 1 \
#       --punct


##########################################################################
# Step 5. Predict source-side trees for GEC training
##########################################################################


# todo:修改数据集
dataset=nlpcc18_translation
CoNLL_SUFFIX=conll_predict_gopar

train_src_file=../../data/${dataset}/trainset/src.txt   # 错误句子文本 ../../data/nlpcc18/trainset/src.txt
train_tgt_file=../../data/${dataset}/trainset/tgt.txt   # 正确句子文本 ../../data/nlpcc18/trainset/tgt.txt
train_src_file_char=../../data/${dataset}/trainset/src.txt.char   # 分词后的错误句子  ../../data/nlpcc18/trainset/src.txt.char
train_tgt_file_char=../../data/${dataset}/trainset/tgt.txt.char   # 分词后的正确句子  ../../data/nlpcc18/trainset/tgt.txt.char

valid_src_file=../../data/${dataset}/validset/src.txt
valid_tgt_file=../../data/${dataset}/validset/tgt.txt
valid_src_file_char=../../data/${dataset}/validset/src.txt.char
valid_tgt_file_char=../../data/${dataset}/validset/tgt.txt.char

test_src_file=../../data/${dataset}/testset/src.txt
test_tgt_file=../../data/${dataset}/testset/tgt.txt
test_src_file_char=../../data/${dataset}/testset/src.txt.char
test_tgt_file_char=../../data/${dataset}/testset/tgt.txt.char

## apply char
## 用 segment_bert.py 来对源文件 $src_file 进行字符级别的分词操作，并将分词结果重定向到 $src_file_char 文件中。

python ../../utils/segment_bert.py <$train_src_file >$train_src_file_char
python ../../utils/segment_bert.py <$train_tgt_file >$train_tgt_file_char
python ../../utils/segment_bert.py <$valid_src_file >$valid_src_file_char
python ../../utils/segment_bert.py <$valid_tgt_file >$valid_tgt_file_char
python ../../utils/segment_bert.py <$test_src_file >$test_src_file_char
python ../../utils/segment_bert.py <$test_tgt_file >$test_tgt_file_char

echo "----- apply char successful !!!!!"

# gopar解析器的路径
parse_path="../../../../gectoolkit/model/SynGEC/src/src_gopar/parse.py"
gopar_path=../../model/gopar/biaffine-dep-electra-zh-gopar
CoNLL_SUFFIX=conll_predict_gopar

echo "gopar start:"
IN_FILE=../../data/${dataset}/trainset/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=3 python ${parse_path} $IN_FILE $OUT_FILE $gopar_path &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"

IN_FILE=../../data/${dataset}/validset/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=3 python ${parse_path} $IN_FILE $OUT_FILE $gopar_path &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"

IN_FILE=../../data/${dataset}/testset/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=3 python ${parse_path} $IN_FILE $OUT_FILE $gopar_path &&\
    echo "Step 5. Predict source-side trees $OUT_FILE successful!"
