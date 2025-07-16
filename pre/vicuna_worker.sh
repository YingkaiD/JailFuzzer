source {conda.sh_PATH}

cd /data/yingkai/autosp/FastChat

conda activate vicuna

CUDA_VISIBLE_DEVICES=2,3 python3 -m fastchat.serve.model_worker --port 23002 --worker-address http://localhost:23002 --controller-address http://localhost:23001 --num-gpus 2  --model-path vicuna-13b-v1.5 --max-gpu-memory 16GiB
