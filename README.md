<div align="center">


<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
  <img src="./figures/logo.png" alt="LUFFY Icon" width="50">
  LUFFY: Learning to Reason Under Off‑Policy Guidance
</h1>

<div align="center">
  <img src="./figures/luffy_intro.jpg" alt="overview" style="width: 66%; height: auto;">
</div>


[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2504.14945)  [![Github](https://img.shields.io/badge/LUFFY-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ElliottYan/LUFFY)   [![Hugging Face Collection](https://img.shields.io/badge/LUFFY_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/yafuly/status/1914559433549676962)

<div align="center" style="font-family: Arial, sans-serif; font-size: 16px;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">🔗 Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a>
  </p>
  <p>
    <a href="#usage" style="text-decoration: none; font-weight: bold;">🔧 Usage</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">📃 Evaluation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a> •
    <!-- <a href="#star-history" style="text-decoration: none; font-weight: bold;">📈 Star History</a> -->
  </p>
</div>



</div>

---

# 🎉News
- **[2025/04/23]** Our paper now available on [alphaXiv](https://www.alphaxiv.org/abs/2504.14945)! We welcome feedback and discussion.
- **[2025/04/23]** 🎉 Ranked **#1** of the day on [Huggingface Daily Papers](https://huggingface.co/papers/2504.14945).
- **[2025/04/20]** The models and datasets are released on [HuggingFace](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4).
- **[2025/04/20]** LUFFY codebase is released along with evaluation scripts. Try it out!
- **[2025/04/20]** LUFFY paper available on [arXiv](http://arxiv.org/abs/2504.14945). 

---

# 🔗Links

- 📜 [Paper (arXiv)](https://arxiv.org/abs/2504.14945)  
- 🤗 [HuggingFace Collection](https://huggingface.co/collections/Elliott/luffy-rl-6804e1f5d1ebe66ba8ac92f4)

---

# ✨Getting Started

## Installation

You can install LUFFY dependencies by running the following commands:
```bash
conda create -n luffy python=3.10
conda activate luffy
cd luffy
pip install -r requirements.txt
pip install -e .
cd verl
pip install -e .
```

If you encounter issues when installing flash-attn, we recommend you to install it here 
[flash-attn](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3). For example, we use this version. 
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Repo Structure

This repository includes:

- `luffy`: Codes for training LUFFY using off-policy reasoning traces. Our main code changes are in luffy/verl/verl/mix_src.
- `data`: Data and code for training and evaluating LUFFY. 
- `exp_scripts`: Example script to train LUFFY.
- `eval_scripts`: Evaluation scripts on math and out-of-distribution benchmarks.

LUFFY is built on top of the GRPO framework and supports plug-and-play integration with off-policy traces from models such as DeepSeek-R1.

---

# 📖Introduction

LUFFY is a reinforcement learning framework that bridges the gap between zero-RL and imitation learning by incorporating off-policy reasoning traces into the training process. Built upon GRPO, LUFFY combines on-policy rollouts with off-policy demonstrations during advantage estimation and introduces **policy shaping** via regularized importance sampling to emphasize low-probability yet crucial actions.

![overview](./figures/luffy_performance.jpg)

### Key Highlights:
- **Off-Policy Guidance:** Seamlessly integrates external reasoning traces to bootstrap learning from stronger models.
- **Dynamic Balance:** Learns when to imitate and when to explore, adapting over the course of training.
- **Policy Shaping:** Emphasizes important actions often ignored in standard policy gradients, enabling better generalization.

---



# 🔧Usage

## Data Preparation
You need to first run the data preparation script to get the training data in parquet format.
```bash
cd data
python prepare_train.py
```

## Training

We provide an example script to train LUFFY on our subset of OpenR1-Math-220k. You can run the following command to train LUFFY:

```bash
  cd exp_scripts
  bash train.sh
```

## Inference

Here’s an example of using LUFFY for inference:

<details>
<summary>Click to view inference example</summary>

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path="Elliott/LUFFY-Qwen-Math-7B-Zero"

question = "which number is larger? 9.11 or 9.9?"

tokenizer = AutoTokenizer.from_pretrained(model_path)
messages = [{"role": "user", "content": question}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_path)
params = SamplingParams(temperature=0.6, max_tokens=8192)
outputs = llm.generate([chat], params)
print(outputs[0].outputs[0].text)
```

</details>


## Models

| **Model**                          | **Huggingface** |  **Base Model** |
|-----------------------------------|------------------|------------------|
| LUFFY-Qwen-Math-7B-Zero | https://huggingface.co/Elliott/LUFFY-Qwen-Math-7B-Zero |  Qwen2.5-Math-7B |
| LUFFY-Qwen-Math-1.5B-Zero | https://huggingface.co/Elliott/LUFFY-Qwen-Math-1.5B-Zero | Qwen2.5-Math-1.5B |
| LUFFY-Qwen-Instruct-7B | https://huggingface.co/Elliott/LUFFY-Qwen-Instruct-7B | Qwen2.5-7B-Instruct |

---

# 📃Evaluation

## LUFFY on Qwen2.5-Math-7B
LUFFY is evaluated on six competition-level benchmarks, achieving state-of-the-art results among all zero-RL methods. It surpasses both on-policy RL and imitation learning (SFT), especially in generalization:



| **Model**                          | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| Qwen2.5-Math-7B                      | 12.9        | 4.2         | 32.6    | 48.8          | 10.7        | 14.8          | 20.7     |
| Qwen2.5-Math-7B-Instruct             | 11.4        | 8.8         | 48.3    | 81.2          | 33.1        | 38.8          | 36.9     |
| SimpleRL-Zero                     | 26.3        | 6.7         | 55.4    | 74.4          | 25.7        | 35.4          | 37.3     |
| OpenReasoner-Zero                 | 17.2        | 15.0        | 52.3    | 84.6          | 33.8        | 47.1          | 41.7     |
| PRIME-Zero                        | 17.9        | 14.7        | 55.2    | 79.4          | **38.2**    | 42.2          | 41.3     |
| Oat-Zero                          | **31.7**    | 11.0        | 61.6    | 79.2          | 29.8        | 42.5          | 42.6     |
| **LUFFY-Qwen-Math-7B-Zero**                         | 29.5        | 23.2        | **66.1**| **88.4**      | 33.8        | **56.4**      | **49.6** |

---



LUFFY also generalizes well to out-of-distribution tasks, with over +6.2 average gain on ARC-C, GPQA, and MMLU-Pro.


| **Model**                         | **ARC-c** | **GPQA-diamond** | **MMLU-Pro** | **Avg.** |
|----------------------------------|-----------|------------------|--------------|----------|
| Qwen2.5-Math-7B             | 18.2      | 11.1             | 16.9         | 15.4     |
| Qwen2.5-Math-7B-Instruct         | 70.3      | 24.7             | 34.1         | 43.0     |
| SimpleRL-Zero                    | 30.2      | 23.2             | 34.5         | 29.3     |
| PRIME-Zero                       | 73.3      | 18.2             | 32.7         | 41.4     |
| Oat-Zero                         | 70.1      | 23.7             | 41.7         | 45.2     |
| OpenReasoner-Zero                | 66.2      | 29.8             | 58.7         | 51.6     |
| **LUFFY-Qwen-Math-7B-Zero**                        | **80.5**    | **39.9**          | **53.0**     | **57.8** |

---

## LUFFY on Qwen2.5-Math-1.5B
| **Model**                          | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| Qwen2.5-Math-1.5B                  |   7.9 | 4.7 | 26.4 | 31.0 | 12.1 | 21.5 | 17.3 |
| Qwen2.5-Math-1.5B-Instruct            |  11.4 | 8.5 | 47.4 | 75.2 | 27.6 | 38.7 | 34.8 |
| **LUFFY-Qwen-Math-1.5B-Zero**             | **15.2** | **12.7** |  **46.8** | **79.4** | **26.5** |  **42.4**  | **42.1** |

## LUFFY on Qwen2.5-Instruct-7B 
| **Model**                          | **AIME 2024** | **AIME 2025** | **AMC** | **MATH-500** | **Minerva** | **Olympiad** | **Avg.** |
|-----------------------------------|-------------|-------------|---------|---------------|-------------|---------------|----------|
| Qwen2.5-7B-Instruct           | 11.9 | 7.6 | 44.1 | 74.6 | 30.5 | 39.7 | 34.7 |
| **LUFFY-Qwen-Instruct-7B**             | **16.6** | **15.7** | **52.2**  | **81.4**  | **36.8**  | **48.7**  | **41.9** |

## Reproducing the Results 
You can reproduce our results by running the following commands:
```bash
ROOT=YOUR_ROOT_PATH
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=Elliott/LUFFY-Qwen-Math-7B-Zero
MODEL_NAME=luffy

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
```

# 🌻Acknowledgement

LUFFY builds upon [veRL](https://github.com/volcengine/verl) and [deepscaler](https://github.com/agentica-project/rllm), and utilizes [vLLM](https://github.com/vllm-project/vllm) for inference. We utilize [Math-Verify](https://github.com/huggingface/Math-Verify) for math reasoning evaluation. We thank the open-source community for datasets and backbones, including [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [DeepSeek-R1](https://github.com/deepseek-ai/deepseek-r1) model. 

# 📬 Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:
- Jianhao Yan: elliottyan37@gmail.com
- Yafu Li: yafuly@gmail.com

# Citation
If you find our model, data, or evaluation code useful, please kindly cite our paper:
```bib
@misc{luffy,
      title={Learning to Reason under Off-Policy Guidance}, 
      author={Jianhao Yan and Yafu Li and Zican Hu and Zhi Wang and Ganqu Cui and Xiaoye Qu and Yu Cheng and Yue Zhang},
      year={2025},
      eprint={2504.14945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.14945}, 
}
```
