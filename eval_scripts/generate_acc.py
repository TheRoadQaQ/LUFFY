#export HF_ENDPOINT=https://hf-mirror.com  
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from math_verify import parse, verify

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    # print(golden_answers)
    # print(predict_answers)
    labels = list(map(verify, golden_answers, predict_answers))
    return labels

def make_conv_zero(question):
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content

def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=8):
    # 数据处理
    df = pd.read_parquet(input_file)

    #df = df[:10]

    num_questions = len(df)
    print(f"there are {num_questions} samples")
    
    messages = df['prompt'].tolist()
    # if debug:
        # messages = messages[:10]
    
    assert remove_system is True
    if remove_system:
        print('remove system')
        assert messages[0][0]['role'] == 'system'
        messages = [message[1:] for message in messages]
    answers = df['reward_model'].tolist()
    answers = [answer['ground_truth'] for answer in answers]
    # if debug:
        # answers = answers[:10]
    assert len(messages) == len(answers)
    data_sources = df['data_source'].tolist()
            
    outputs = generate_vllm(messages, model_path, template=template, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)

    print(len(outputs),num_questions)

    accs = []
    generated_texts_list = []

    print(outputs[0])

    for i in range(num_questions):
        correct_count = 0
        
        output = outputs[i]
        answer = answers[i]
        
        generated_texts = []
        prompt = output.prompt
            
        for j in range(n):  # 遍历这个问题的n个回答
            generated_text = output.outputs[j].text
            
            if prompt.endswith(THOUGHT_DELIMITER_START+'\n'):
                generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
                
            if THOUGHT_DELIMITER_START in generated_text and THOUGHT_DELIMITER_END in generated_text:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            
            labels = labeling_responses([generated_text,], answer)
            
            generated_texts.append({
                'generated_text': generated_text,
                'correctness': labels[0]
            })
            
            if labels[0]:
                correct_count += 1
    
        # 计算当前组的准确率并添加到acc_list
        group_acc = correct_count / n
        accs.append(group_acc)
        generated_texts_list.append(generated_texts)

    df["accs"] = accs
    df["answers"] = generated_texts_list

    df.to_parquet(output_file)
            

def generate_vllm(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=8):
    #vllm模型加载
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=n ,temperature=temperature, top_p=top_p, max_tokens=8192)
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count()
             )  # 替换成本地路径

    gen_prompts = []
    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own': 
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True
        )
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[0]['content'])
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[0]['content'])
        elif template == 'no':
            gen_prompt = cur_message[0]['content']
        else: raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)
        if i == 0:
            print('Example input: ', gen_prompt)

    outputs = llm.generate(gen_prompts, sampling_params)
    return outputs

if __name__ == "__main__":
    import fire
    fire.Fire(main)