import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载目标模型和草稿模型
target_model_name = "Qwen/Qwen2.5-7B-Instruct"
draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
target_model = AutoModelForCausalLM.from_pretrained(target_model_name, device_map="auto", trust_remote_code=True)
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, device_map="auto", trust_remote_code=True)

# 设置模型为评估模式
target_model.eval()
draft_model.eval()

# 定义speculative decoding函数
def speculative_decoding(prompt, max_length=50, alpha=2):
    # 使用草稿模型生成初始序列
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(target_model.device)
    #input_ids: [1, 6]
    with torch.no_grad():
        draft_outputs = draft_model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
    #draft_outputs.sequences: [1, 50], 109924, 104133,...

    # 获取草稿模型生成的token序列
    draft_tokens = draft_outputs.sequences[0]
    #draft_tokens: [50], 109924, 104133,...
    draft_token_scores = draft_outputs.scores

    # 初始化最终输出序列
    final_output_ids = input_ids[0]
    #final_output_ids: [6]

    # 逐步验证草稿模型生成的token
    i = 0
    #用i表示当前draft相对位置
    while i < len(draft_tokens) - len(input_ids[0]):
        #50 - 6
        print('当前预测绝对位置索引:', len(input_ids[0]) + i)
        print('当前预测draft相对位置索引i:', i)
        print('当前prompt长度length of input_ids[0]:', len(input_ids[0]))
        if len(input_ids[0]) + i >= len(draft_tokens):
            print(f"Index {len(input_ids[0]) + i} is out of bounds for draft_tokens with size {len(draft_tokens)}")
            break
        token_id = draft_tokens[len(input_ids[0]) + i].unsqueeze(0)
        #print('token_id', token_id)
        #token_id: [1]，len(input_ids[0]) + i表示当前绝对位置

        # 计算目标模型的token概率
        with torch.no_grad():
            target_outputs = target_model(final_output_ids.unsqueeze(0))
            target_logits = target_outputs.logits[:, -1, :]
            #target_logits:[B,V]
            #target_outputs.logits:[B,S,V]
            target_probs = torch.softmax(target_logits, dim=-1)
            #target_probs:[B,V]
        
        # 获取草稿模型的token概率
        draft_prob = torch.softmax(draft_token_scores[i], dim=-1)
        #draft_prob:[B,S]

        # 计算加速比alpha，决定是否接受草稿模型的token
        #print('target_probs[0, token_id]:', target_probs[0, token_id])
        #print('draft_prob[0, token_id]:', draft_prob[0, token_id])
        acceptance_ratio = target_probs[0, token_id] / (alpha * draft_prob[0, token_id])

        if acceptance_ratio >= 1:
            # 接受草稿模型的token
            print('!!!接受!!!')
            final_output_ids = torch.cat([final_output_ids, token_id])
            #final_output_ids: [x]
            i += 1

        else:
            print('!!!拒绝!!!')
            # 使用目标模型采样下一个token
            with torch.no_grad():
                target_next_token = torch.multinomial(target_probs, num_samples=1)
            final_output_ids = torch.cat([final_output_ids, target_next_token.squeeze(0)])

            # 更新草稿模型的输入
            input_ids = final_output_ids.unsqueeze(0)
            #input_ids: [1, x]
            i = 0
        

    # 解码最终的输出序列
    generated_text = tokenizer.decode(final_output_ids, skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "从前有一个年轻的程序员，他"
    output_text = speculative_decoding(prompt)
    print("生成的文本：")
    print(output_text)