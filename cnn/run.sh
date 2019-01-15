start_time=`date +"%Y-%m-%d %H:%M:%S"`
echo "start running at "$start_time

CUDA_VISIBLE_DEVICES=1 python cnn.py --dataset mpqa --topk 50 --batchsize 64 --experiment 1

end_time=`date +"%Y-%m-%d %H:%M:%S"`
echo "Finish at "$end_time
