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

def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, prefix_rate=0.5):
    # 数据处理
    df = pd.read_parquet(input_file)
    messages = df['prompt'].tolist()
    
    assert remove_system is True
    if remove_system:
        print('remove system')
        assert messages[0][0]['role'] == 'system'
        messages = [message[1:] for message in messages]
    answers = df['reward_model'].tolist()
    answers = [answer['ground_truth'] for answer in answers]
    
    assert len(messages) == len(answers)

    tgt_solutions = df['target'].tolist()
    tgt_solutions = [solution[0]["content"] for solution in tgt_solutions]
    tgt_solutions = [solution[len('<think>\n'):] for solution in tgt_solutions if solution.startswith('<think>\n')]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=8192)
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count())

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
        else:
            raise ValueError(f'Invalid template: {template}')
        
        gen_prompt += tgt_solutions[i][:int(len(tgt_solutions) * prefix_rate)]
        
        gen_prompts.append(gen_prompt)
    
    outputs = llm.generate(gen_prompts, sampling_params)
    
    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg = 0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        answer = answers[i]
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n'):
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            
        if THOUGHT_DELIMITER_START in generated_text and THOUGHT_DELIMITER_END in generated_text:
            generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
        
        # try:
        labels = labeling_responses([generated_text,], answer)
        
        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0]
        })
        if labels[0]:
            avg += 1
            
    print('accuracy: ', avg / len(outputs))
    
    #for data_source, labels in rets.items():
    # print(data_source, len(labels))
    #acc = np.array(labels).mean()
    #print(f'{data_source}: {acc}')
    
    with open(output_file, 'w') as f:
        for item in save_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    import fire
    fire.Fire(main)