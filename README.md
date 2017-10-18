# FPN
-----------------


end2end testing:
[mAP(0.7832)](https://drive.google.com/open?id=0B_qzepxA9F3vbDRnT1JoNjZtekU)
without bells and whistles, without OHEM

CUDA_VISIBLE_DEVICES=0 python ./faster_rcnn/test_net.py --gpu 0 --weights
output/FPN_end2end/voc_0712_trainval/FPN_iter_370000.ckpt
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
1. imporve end2end training result
2. check roi_pooling used interpolation or not
3. fix bugs in alt_opt training
