# Mini-Llama-Chinese
<p align="center">
  <img src="Mini-Llama2-Chinese.png" alt="Mini-Llama-Chinese" style="width:30%;">
</p>

想要从零开始训练一个中文的mini大语言模型，目前的目标是学习和尝试使用各种方法训练和优化模型，最终训练一个较小的模型可以进行基本的对话，具体模型大小根据手头的机器决定。目前模型训练代码主要参考[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)，只是修改了部分参数，这里就不放代码了，后面如果对模型代码有较大的更改再上传。   

**目前还在不断的学习和尝试中，后续会将训练好的模型整理，上传。大家有什么可以提高模型效果的建议可以留言，会选择进行尝试优化，训练完成后再上传上来，一起试试效果。**
|模型名称|训练数据|模型下载地址|
|:----:|:----:|:----:|
|[model-0](#model-0)|维基百科中文、百度百科、医疗文本|[模型下载](https://huggingface.co/My521/Mini-Llama-Chinese)|
|[model-0-sft](#model-0)|bell、alpaca-zh|[模型下载](https://huggingface.co/My521/Mini-Llama-Chinese) |  
|[model-1](#model-1)|英文文本108G，中文文本217G，中英翻译文本6G| |
|[model-1-sft](#model-1)| | | 
|[model-2](#model-2)|英文文本318G，中文文本232G| |
|[model-2-sft](#model-2)| |  | 
|[model-3](#model-3)| 英文文本318G，中文文本232G| |
|[model-3-sft](#model-3)| |  | 


### model-0  
模型预训练数据：维基百科中文、百度百科、医疗文本，共17GB的文本，token数量没有计算。  
预训练轮数：1 epoch。
模型结构：max_seq_len = 512，dim = 1024，n_layers = 12，n_heads = 8。  
指令微调数据：直接使用了[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)、[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)的微调数据集，没有进行清洗。  
指令微调轮数：2 epoch。  
模型效果：这里随机抽取了[firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)指令微调数据集测试，[测试结果](https://huggingface.co/My521/Mini-Llama-Chinese/tree/main/model0)只能说是啥也不会，只是能生成汉字，语言随机，像一个刚出生的婴儿，慢慢成长吧。


### model-1  
模型预训练数据：英文文本108G，中文文本217G，中英翻译文本6G，token数量没有计算。  
预训练轮数：1 epoch。
模型结构：max_seq_len = 512，dim = 1024，n_layers = 16，n_heads = 16。  
指令微调数据：    
指令微调轮数：  
模型效果：  

### model-2  
模型预训练数据：英文文本318G，中文文本232G，token数量中文58 B，英文81 B，共约140 B token。  
预训练轮数：1 epoch。
模型结构：max_seq_len = 512，dim = 1024，n_layers = 16，n_heads = 16。  
指令微调数据：    
指令微调轮数：  
模型效果：  

### model-3  
模型预训练数据：与model-2相同。  
预训练轮数：1 epoch。
模型结构：max_seq_len = 512，dim = 1024，n_layers = 32，n_heads = 16。  
指令微调数据：    
指令微调轮数：  
模型效果：  