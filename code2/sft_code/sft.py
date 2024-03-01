
import logging
import numpy as np
import os
import glob
import json
import sys
import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import pandas as pd
import datasets
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, concatenate_datasets,Dataset
from datetime import datetime, timezone
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from model import Transformer, ModelArgs
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from dataset_sft import SFTDataset


def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)




MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )


    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class MyTrainingArguments(TrainingArguments):
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    use_flash_attention_2 : Optional[bool] = field(default=True)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )



    set_seed(training_args.seed)



    #############
    #model = AutoModelForCausalLM.from_config(config)
    #########################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device = 'cuda'
    init_from = 'scratch'
    # max_seq_len = 512
    # dim = 2048
    # n_layers = 32
    # n_heads = 32
    # multiple_of = 32
    # dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
    # bias = False # do we use bias inside LayerNorm and Linear layers?
    # def init_model():
    # # model init

    #     model_args = dict(
    #         dim=dim,
    #         n_layers=n_layers,
    #         n_heads=n_heads,
    #         n_kv_heads=n_heads,
    #         vocab_size=64798,
    #         multiple_of=multiple_of,
    #         max_seq_len=max_seq_len,
    #         dropout=dropout,
    #     )  # start with model_args from command line
    #     if init_from == "scratch":
    #         # init a new model from scratch
    #         print("Initializing a new model from scratch")
    #         gptconf = ModelArgs(**model_args)
    #         model = Transformer(gptconf)
    #     return model
    # model=init_model()
    
    
    device = 'cuda'
    init_from = 'scratch'
    def load_config(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    print(f"-----------model_args.config_name----------:{model_args.config_name}")
    config = load_config(model_args.config_name)
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
    state_dict = torch.load('./pretrain_code/output_dir/checkpoint-2882583/pytorch_model.bin')

    # 创建一个新的状态字典，去除 _orig_mod 前缀
    new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

    # 加载新的状态字典到模型中
    model.load_state_dict(new_state_dict)
    model.to(device)
    model = torch.compile(model)
    ################
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Total number of trainable parameters: {n_params:,}")
    
    
    
    prompt = []
    answer=[]
    with open('./sft_data.json','r') as file:
        for line in file:
            item = json.loads(line)
            prompt.append(item['question'])
            answer.append(item['answer'])
    df = pd.DataFrame()   
    df['prompt']=prompt
    df['answer']=answer    
         
    df=df.sample(frac=1.0)
    print(df)
    tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    train_ds = SFTDataset(df,tokenizer, max_length=512)


    
    def my_data_collator(examples):
        # 将所有样本的输入 (`X`) 和标签 (`Y`) 分别堆叠
        input_ids = torch.stack([example[0] for example in examples])
        labels = torch.stack([example[1] for example in examples])

        # 返回一个字典，包含模型需要的键和值
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=my_data_collator,
    )
    # Training
    trainer.train()
    
    


if __name__ == "__main__":
    main()
