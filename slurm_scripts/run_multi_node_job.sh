# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# A script with a list of commands for submitting SLURM jobs

#### Kinetics training
JOB_NAME=TimeSformer_divST_8x32_224
python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition dev --comment "" --name ${JOB_NAME} --use_volta32

#JOB_NAME=TimeSformer_jointST_8x32_224
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_jointST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

#JOB_NAME=TimeSformer_spaceOnly_8x32_224
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

#### Kinetics inference
#JOB_NAME=TimeSformer_divST_8x32_224_TEST_3clips
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml --job_dir /your/job/dir/${JOB_NAME}/  --num_shards 4 --partition dev --comment "" --name ${JOB_NAME} --use_volta32


##### SSv2 training
#JOB_NAME=TimeSformer_divST_8_224
#python tools/submit.py --cfg configs/SSv2/TimeSformer_divST_8_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

##### Sth-Sth_v2 inference
#JOB_NAME=TimeSformer_divST_8_224_TEST_3clips
#python tools/submit.py --cfg configs/SSv2/TimeSformer_divST_8_224_TEST.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32
