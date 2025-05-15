#export HF_ENDPOINT=https://hf-mirror.com  
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from collections import defaultdict

from math_verify import parse, verify

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def calculate_token_probabilities(output, tokenizer):
    """计算生成文本中每个token的概率"""
    token_probs = []
    token_texts = []
    
    # 获取生成的token IDs和对应的logprobs
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # 将logprobs转换为概率
    for token_id, logprob in zip(token_ids, logprobs):
        prob = np.exp(logprob) if logprob is not None else 1.0
        token_text = tokenizer.decode([token_id])
        token_probs.append(prob)
        token_texts.append(token_text)
    
    return token_texts, token_probs

def analyze_answer_probability(prompt, answer, model, tokenizer, sampling_params):
    """分析给定答案在模型下的token概率"""
    # 生成完整prompt+answer的token概率
    full_text = prompt + answer
    outputs = model.generate([full_text], sampling_params)
    output = outputs[0].outputs[0]
    
    # 计算所有token的概率
    all_tokens, all_probs = calculate_token_probabilities(output, tokenizer)
    
    # 只提取answer部分的概率
    prompt_tokens = tokenizer.encode(prompt)
    answer_tokens = tokenizer.encode(answer)
    
    # 找到answer tokens在完整输出中的位置
    answer_start = len(prompt_tokens)
    answer_end = answer_start + len(answer_tokens)
    
    answer_token_texts = all_tokens[answer_start:answer_end]
    answer_token_probs = all_probs[answer_start:answer_end]
    
    return answer_token_texts, answer_token_probs

def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192):
    # 数据处理
    df = pd.read_parquet(input_file)
    messages = df['prompt'].tolist()
    solutions = df['target'].tolist()
    
    assert len(messages) == len(solutions)
    
    # 初始化模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        max_tokens=max_tokens,
        logprobs=1  # 确保返回logprobs
    )
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count())
    
    rets = defaultdict(list)
    save_data = []
    avg = 0
    
    for i, (message, solution) in enumerate(zip(messages, solutions)):
        # 生成prompt
        if template == 'own': 
            gen_prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(message[0]['content'])
        elif template == 'prime':
            gen_prompt = make_conv_zero(message[0]['content'])
        elif template == 'no':
            gen_prompt = message[0]['content']
        else: 
            raise ValueError(f'Invalid template: {template}')
        
        # 计算答案token概率
        answer_tokens, answer_probs = analyze_answer_probability(
            gen_prompt, answer, llm, tokenizer, sampling_params
        )
        
        # 生成模型自己的回答
        outputs = llm.generate([gen_prompt], sampling_params)
        output = outputs[0].outputs[0]
        generated_text = output.text
        
        # 计算生成文本的token概率
        gen_tokens, gen_probs = calculate_token_probabilities(output, tokenizer)
        
        # 评估正确性
        labels = labeling_responses([generated_text,], answer)
        
        rets[data_sources[i]].append(labels[0])
        
        save_data.append({
            'prompt': gen_prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0],
            'answer_token_probs': answer_probs,
            'answer_tokens': answer_tokens,
            'generated_token_probs': gen_probs,
            'generated_tokens': gen_tokens
        })
        
        if labels[0]:
            avg += 1
            
        # 调试输出
        if debug and i < 3:
            print(f"\nSample {i}:")
            print(f"Prompt: {gen_prompt[:100]}...")
            print(f"Answer: {answer}")
            print(f"Answer tokens: {answer_tokens}")
            print(f"Answer probs: {answer_probs}")
            print(f"Generated: {generated_text[:100]}...")
            print(f"Generated tokens: {gen_tokens[:10]}...")
            print(f"Generated probs: {gen_probs[:10]}...")
    
    print('accuracy: ', avg / len(messages))
    
    for data_source, labels in rets.items():
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')
    
    try:
        with open(output_file, 'w') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')

if __name__ == "__main__":
    import fire
    fire.Fire(main)