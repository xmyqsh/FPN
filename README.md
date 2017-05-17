# FPN
-----------------
nohup ./experiments/scripts/FPN_end2end.sh 1 FPN pascal_voc0712 --set
RNG_SEED 42 TRAIN.SCALES "[800]" > FPN.log 2>&1 &

tail -f FPN.log
