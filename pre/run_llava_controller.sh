source {conda.sh_PATH}

cd {LLAVA_PATH}

conda activate llava

python -m llava.serve.controller --host 0.0.0.0 --port 10000
