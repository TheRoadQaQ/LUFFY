from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DataParallel

def batch_calculate_probs_batch(questions, answers):
    # 编码每个样本的question和answer
    batch_question_ids = []
    batch_answer_ids = []
    
    for q, a in zip(questions, answers):
        q_tensor = tokenizer.encode(q, add_special_tokens=False, return_tensors="pt")[0]
        a_tensor = tokenizer.encode(a, add_special_tokens=False, return_tensors="pt")[0]
        batch_question_ids.append(q_tensor)
        batch_answer_ids.append(a_tensor)
    
    full_ids = [torch.cat([q, a]) for q, a in zip(batch_question_ids, batch_answer_ids)]
    padded_full_ids = pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (padded_full_ids != tokenizer.pad_token_id).to("cuda")
    
    inputs = padded_full_ids.to("cuda")
    
    with torch.no_grad():
        outputs = model(inputs, attention_mask=attention_mask)
    
    batch_results = []
    for i in range(len(questions)):
        q_len = len(batch_question_ids[i])
        a_len = len(batch_answer_ids[i])
        
        start_pos = q_len - 1
        logits = outputs.logits[i, start_pos:start_pos+a_len]
        answer_tokens = batch_answer_ids[i].to(logits.device)
        
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[torch.arange(a_len, device=logits.device), answer_tokens].to("cpu")
        
        tokens = tokenizer.convert_ids_to_tokens(batch_answer_ids[i])
        batch_results.append(list(zip(tokens, token_probs.tolist())))

    del outputs,logits, probs, token_probs
    torch.cuda.empty_cache()
    
    return batch_results

# 模型加载
model_name = "/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用DataParallel包装模型
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
model = DataParallel(model, device_ids=list(range(8)))  # 使用全部8块GPU
model.eval()

# 数据加载
df = pd.read_parquet("../dataset/sub_1000_openr1.parquet")
results = []
batch_size = 8  # 调整为原来的8倍（8卡 * 原单卡batch_size4）

from tqdm import tqdm

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    processed_questions = []
    processed_answers = []
    
    for _, row in batch.iterrows():
        prompt = tokenizer.apply_chat_template(
            row['prompt'],
            tokenize=False,
            add_generation_prompt=True
        )
        
        target = row['target'][0]["content"]
        if prompt.endswith('<think>\n') and target.startswith('<think>\n'):
            target = target[len('<think>\n'):]
        
        processed_questions.append(prompt)
        processed_answers.append(target)
    
    batch_results = batch_calculate_probs_batch(processed_questions, processed_answers)
    results.extend(batch_results)

np.save('./results/sub_1000_probs_results.npy', np.array(results, dtype=object), allow_pickle=True)