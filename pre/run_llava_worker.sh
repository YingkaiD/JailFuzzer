source {conda.sh_PATH}

cd {LLAVA_PATH}

conda activate llava

CUDA_VISIBLE_DEVICES=4,5 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path model_trained/llava-v1.5-13b
