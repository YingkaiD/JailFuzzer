# JailFuzzer

This if the official implementation for paper: [Fuzz-Testing Meets LLM-Based Agents: An Automated and Efficient Framework for Jailbreaking Text-to-Image Generation Models](https://arxiv.org/abs/2408.00523)

## Environment setup

```
conda create -n jailfuzzer python=3.9
pip install -r requirements.txt
```

## Dataset

This project uses exactly the same dataset as the [SneakyPrompt](https://github.com/Yuchen413/text2image_safety/tree/main).


## Running

Before running the main program, you need to run the core model of the Mutation Agent (e.g. LLaVA), the core model of the Oracle Agent (e.g. Vicuna), word embedding tool (e.g. SentenceTransformer), and target T2I model (e.g. sd14).

### Run LLaVA

Please follow the [LLaVA GitHub page](https://github.com/haotian-liu/LLaVA/) to install LLaVA.

```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

In one terminal, start the controller first:
```
bash pre/run_llava_controller.sh
```
Then, in another terminal, start the worker:
```
bash pre/run_llava_worker.sh
```

### Run Vicuna

Please follow the [FastChat GitHub page](https://github.com/lm-sys/FastChat) to install Vicuna.
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat

conda create -n vicuna python=3.10 -y
conda activate vicuna

pip3 install --upgrade pip
pip3 install -e ".[model_worker,webui]"
```

In one terminal, start the controller first:
```
bash pre/vicuna_controller.sh
```
Then, in another terminal, start the worker:
```
bash pre/vicuna_worker.sh
```

### Run SentenceTransformer

```
conda create -n st python=3.10 -y
pip install -U sentence-transformers
pip install fastapi uvicorn
```

In one terminal, start SentenceTransformers:
```
bash pre/st_api.sh
```

### Run Stable Diffusion v1.4 and Text-Image-Based Safety Filter

```
conda create -n sd14 python=3.10 -y
pip install --upgrade diffusers transformers scipy
pip install fastapi uvicorn

python t_i.py
```

### Run JailFuzzer
```
conda activate jailfuzzer

python main.py
```

## Citation:

Please cite our paper if you find this repo useful.

```
@inproceedings{dong2025fuzz,
  title={Fuzz-testing meets llm-based agents: An automated and efficient framework for jailbreaking text-to-image generation models},
  author={Dong, Yingkai and Meng, Xiangtao and Yu, Ning and Li, Zheng and Guo, Shanqing},
  booktitle={2025 IEEE Symposium on Security and Privacy (SP)},
  pages={373--391},
  year={2025},
  organization={IEEE}
}
```
