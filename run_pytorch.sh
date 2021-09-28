# ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
# docker pull ${ImageName}

# run_cmd="cd /home;
#          git clone https://github.com/HydrogenSulfate/TimeSformer.git
#          cd TimeSformer;
#          CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 1 fp32 TimeSformer;
#          CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh mp 1 fp32 TimeSformer;
#          "

# run_cmd="cd /home;
#          git clone https://github.com/HydrogenSulfate/TimeSformer.git
#          cd TimeSformer;
#          CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 1 fp32 TimeSformer;
#          CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh mp 1 fp32 TimeSformer;
#          "
model_mode=TimeSformer
run_mode=sp
bs=1
fp_item=fp32
num_gpu_devices=1

CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp ${num_gpu_devices} fp32 TimeSformer
sleep 60
bs=1
python3.7 ../PaddleVideo/scripts/analysis.py \
        --filename ${model_mode}_${run_mode}_bs`expr ${bs} / ${num_gpu_devices}`_${fp_item}_${num_gpu_devices} \
        --keyword "avg_ips:" \
        --model_name video_${model_mode}_bs`expr ${bs} / ${num_gpu_devices}`_${fp_item} \
        --mission_name "action recognition" \
        --direction_id 0 \
        --run_mode ${run_mode} \
        --gpu_num ${num_gpu_devices} \
        --index 1

run_mode=mp
bs=4
fp_item=fp32
num_gpu_devices=4

CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh mp `expr ${bs} / ${num_gpu_devices}` fp32 TimeSformer;
sleep 60
python3.7 ../PaddleVideo/scripts/analysis.py \
        --filename ${model_mode}_${run_mode}_bs`expr ${bs} / ${num_gpu_devices}`_${fp_item}_${num_gpu_devices} \
        --keyword "avg_ips:" \
        --model_name video_${model_mode}_bs`expr ${bs} / ${num_gpu_devices}`_${fp_item} \
        --mission_name "action recognition" \
        --direction_id 0 \
        --run_mode ${run_mode} \
        --gpu_num ${num_gpu_devices} \
        --index 1


# nvidia-docker run --name test_torch_seg -it  \
#     --net=host \
#     --shm-size=1g \
#     -v $PWD:/workspace \
#     ${ImageName}  /bin/bash -c "${run_cmd}"

# nvidia-docker stop test_torch_seg
# nvidia-docker rm test_torch_seg 