source {conda.sh_PATH}

cd {FASTCHAT_PATH}

conda activate vicuna

python3 -m fastchat.serve.controller --host 0.0.0.0 --port 23001

