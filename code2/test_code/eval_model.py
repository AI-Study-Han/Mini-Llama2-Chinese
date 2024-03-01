
import torch
import json
from model import Transformer, ModelArgs
import os
from contextlib import nullcontext
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

device = 'cuda'
init_from = 'scratch'
dtype = "float32"
max_new_tokens = 512 
temperature = 0.8
top_k = 30

device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
config = load_config('./model_config.json')
print(config)
def init_model(config):
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**config)
        model = Transformer(gptconf)
    return model
model=init_model(config)

    # 加载模型权重
state_dict = torch.load('./output_dir/sft/checkpoint-28701/pytorch_model.bin')

# 创建一个新的状态字典，去除 _orig_mod 前缀
new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

# 加载新的状态字典到模型中
model.load_state_dict(new_state_dict)
model.to(device)
tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
my_list = [
    "你知道北京吗？",
    "你知道杭州有哪些美食吗？",
    "你知道中国的四大名著吗？",
    "你了解美国的历史吗？",
    "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
    "鸡兔同笼，共35只头，94只脚，问鸡兔各多少？",
    "将以下中文翻译成英文：穿衣需要适应天气变化。"
]
for question in my_list:
    x=tokenizer.encode(question,add_special_tokens=False)+[tokenizer.special_tokens['<bos>']]
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 2, max_new_tokens, temperature=temperature, top_k=top_k)
            #
            answer=tokenizer.decode(y[0].tolist())
            answer=answer.replace(question,'')

            print('[question]:',question)
            print('[answer]:',answer)
            print('---------------')