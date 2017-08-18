# FPN
-----------------

alt_opt training:

nohup ./experiments/scripts/FPN_alt_opt.sh 0 FPN_alt_opt pascal_voc0712
--set RNG_SEED 42 TRAIN.SCALES "[800]" > FPN_alt_opt.log 2>&1 &



end2end training:

nohup ./experiments/scripts/FPN_end2end.sh 1 FPN pascal_voc0712 --set
RNG_SEED 42 TRAIN.SCALES "[800]" > FPN.log 2>&1 &

tail -f FPN.log


------------------------

TODO:
1. Testing result
2. acceleration by changing fake fpn_roi_pooling to real tf op
