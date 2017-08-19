# FPN
-----------------


end2end testing:
[mAP(0.7372)](https://drive.google.com/open?id=0B_qzepxA9F3vbDRnT1JoNjZtekU)
Further improvement may be done by decrease the BATCH_SIZE and learning
rate for stability of loss curve

python ./faster_rcnn/test_net.py --gpu 0 --weights
output/FPN_end2end/voc_0712_trainval/FPN_iter_300000.ckpt
--imdb voc_0712_test --cfg ./experiments/cfgs/FPN_end2end.yml --network
FPN_test



alt_opt training:

nohup ./experiments/scripts/FPN_alt_opt.sh 0 FPN_alt_opt pascal_voc0712
--set RNG_SEED 42 TRAIN.SCALES "[800]" > FPN_alt_opt.log 2>&1 &



end2end training:

nohup ./experiments/scripts/FPN_end2end.sh 1 FPN pascal_voc0712 --set
RNG_SEED 42 TRAIN.SCALES "[800]" > FPN.log 2>&1 &

tail -f FPN.log


------------------------

TODO:
1. imporve end2end training result, test alt_opt training result
2. acceleration by changing fake fpn_roi_pooling to real tf op
