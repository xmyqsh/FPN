#!/bin/bash
#
# Example:
# ./experiments/scripts/FPN_alt_opt.sh 1 FPN pascal_voc0712 \
#   --set RNG_SEED 42 TRAIN.SCALES "[800]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc0712)
    TRAIN_IMDB="voc_0712_trainval"
    TEST_IMDB="voc_0712_test"
    PT_DIR="pascal_voc"
    CFG="experiments/cfgs/FPN_alt_opt.yml"
    ;;
  pascal_voc2007)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=200000
    CFG="experiments/cfgs/FPN_alt_opt.yml"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac
LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y_%m_%d_%H_%M_%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/train_net_alt_opt.py --gpu ${GPU_ID} \
  --weights data/pretrain_model/Resnet50.npy \
  --imdb ${TRAIN_IMDB} \
  --cfg ${CFG} \
  --network FPN_alt_opt_train_test \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $6}'`
#NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $3}'`
echo $NET_FINAL
set -x

time python ./tools/test_net.py --gpu ${GPU_ID} \
  --weights ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg ${CFG} \
  --network FPN_alt_opt_train_test
