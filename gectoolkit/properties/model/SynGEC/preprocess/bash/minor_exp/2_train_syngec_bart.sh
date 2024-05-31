####################
# Train Baseline
####################

SEED=2023


dataset='cowsl2h'
language='spanish'

MODEL_DIR_STAGE1=../../model/${language}_bart_baseline/$SEED/stage1
PROCESSED_DIR_STAGE1=../../preprocess/${language}_${dataset}_with_syntax_bart

FAIRSEQ_CLI_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
FAIRSEQ_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq

BART_PATH=../../pretrained_weights/bart.large/pytorch_model.bin

mkdir -p $MODEL_DIR_STAGE1

mkdir -p $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE1/src

cp ./train_syngec_bart.sh $MODEL_DIR_STAGE1

# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=6 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
    --save-dir $MODEL_DIR_STAGE1 \
    --arch bart_large \
    --restore-file $BART_PATH \
    --task translation \
    --max-tokens 5120 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq 1 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler polynomial_decay \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --patience 10 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 10 \
    --seed $SEED 2>&1 | tee ${MODEL_DIR_STAGE1}/nohup.log
# --seed $SEED >${MODEL_DIR_STAGE1}/nohup.log 2>&1 &

wait

#####################
## Train SynGEC
#####################

SEED=2022
FAIRSEQ_CLI_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq_cli

# todo:修改
MODEL_DIR_STAGE1=../../model/english_fce_bart_syngec/
#conll14：PROCESSED_DIR_STAGE1=../../preprocess/english_conll14_with_syntax_bart
PROCESSED_DIR_STAGE1=../../preprocess/english_fce_with_syntax_bart

FAIRSEQ_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq

#BART_PATH=../../model/english_bart_baseline/$SEED/stage3/checkpoint_best.pt
BART_PATH=../../download_checkpoint/english_bart_baseline.pt

#mkdir -p $MODEL_DIR_STAGE1
#mkdir -p $MODEL_DIR_STAGE1/src
#cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src
#cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src
#cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE1/src
#cp ./2_train_syngec_bart.sh $MODEL_DIR_STAGE1

# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=7 python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
    --save-dir $MODEL_DIR_STAGE1 \
    --user-dir ../../src/src_syngec/syngec_model \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --freeze-bart-parameters \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_bart_large \
    --restore-file $BART_PATH \
    --max-sentence-length 64 \
    --max-tokens 1024 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq 8 \
    --lr 5e-04 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler polynomial_decay \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --patience 10 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 10 \
#    --seed $SEED >${MODEL_DIR_STAGE1}/nohup.log 2>&1 &


