import argparse
import torch
import numpy as np
import torch.nn.functional as F
import wandb
import random

from tqdm import tqdm
from peft import get_peft_model, TaskType, PeftConfig, PeftModel
from peft import PromptTuningConfig, PromptTuningInit
from peft import LoraConfig, PrefixTuningConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import default_data_collator, get_linear_schedule_with_warmup
from datasets import Dataset
from torch.utils.data import DataLoader

from typing import Union, Callable, Tuple
from collections import namedtuple
from src.data import load_mcrc_data

#== General Util tools ============================================================================#
CAUSAL_LLMS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "gpt2": "gpt2"
}

SEQ2SEQ_LLMS = {}

MODEL_NAME_TO_PATH = {**CAUSAL_LLMS, **SEQ2SEQ_LLMS}
MODEL_PATH_TO_NAME = {v:k for k, v in MODEL_NAME_TO_PATH.items()}

#== Model utility functions ======================================================================#
def is_seq2seq(model_path):
    if model_path in SEQ2SEQ_LLMS.values():
        output = True
    elif model_path in CAUSAL_LLMS.values():
        output = False
    else:
        raise ValueError('invalid model name')
    return output

#== Data Preprocessing ============================================================================#
def process_mcrc_dataset(dataset_name:str, model_name:str):
    prompt_template = '{context}\n\n{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}'

    if not is_seq2seq(model_name):
        prompt_template = prompt_template + '\n\nAnswer:'

    # load data
    train, val, test = load_mcrc_data(dataset_name)

    # process data
    LABEL_WORDS = ['A', 'B', 'C', 'D']
    train_data = []
    val_data = []
    test_data = []

    for split, output in [(train, train_data), (val, val_data), (test, test_data)]:
        for ex in split:
            input_text = prompt_template.format(
                context=ex.context, 
                question=ex.question,
                option_1=ex.options[0],
                option_2=ex.options[1],
                option_3=ex.options[2],
                option_4=ex.options[3]
            )
            label_text = LABEL_WORDS[ex.label]
            output.append(({'text':input_text, 'label_text':label_text, 'label_id':ex.label}))

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    return train_dataset,  val_dataset, test_dataset

def create_train_preprocess_function(tokenizer, max_length)->Callable[[dict], dict]:
    def preprocess_function(examples):
        inputs = [x for x in examples['text']]
        targets = [x for x in examples['label_text']]
        batch_size = len(inputs)

        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)  # don't add bos since concatenation
        
        #tokenize input and label text, and prepare for training
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100]*len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1]*len(model_inputs["input_ids"][i])
        
        #truncating and padding to max length
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            
            # do padding
            pad_length = max_length-len(sample_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id]*pad_length
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [0]*pad_length
            labels["input_ids"][i] = label_input_ids + [-100]*pad_length
            
            # truncate
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        
        model_inputs["labels"] = labels["input_ids"]       
        return model_inputs
    return preprocess_function

def create_eval_preprocess_function(tokenizer, max_length=None)->Callable:
    def preprocess_function(examples):
        inputs = [x for x in examples['text']]
        targets = [x for x in examples['label_id']]
        batch_size = len(inputs)

        model_inputs = tokenizer(inputs)
        
        #truncating and padding to max length
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            
            if max_length:
                # do padding
                pad_length = max_length-len(sample_input_ids)
                model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id]*pad_length
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [0]*pad_length
                
                # truncate
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            else:
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])

        model_inputs["label"] = torch.tensor(targets)
        return model_inputs
    return preprocess_function

def tokenize_dataset(dataset:Dataset, preprocess_function:Callable, num_proc:int=10)->Tuple[Dataset]:
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.column_names,
    )
    return processed_dataset

def filter_dataset(tok_dataset:Dataset, max_length:int=None, size:int=None, num_proc:int=10)->Dataset:
    #remove inputs that are too long
    if max_length:
        filter_fn = lambda example: not all([x == -100 for x in example['labels']])
        tok_dataset = tok_dataset.filter(filter_fn, num_proc=num_proc)
    
    #reduce traing set if shorter sizes provided
    if size and size < len(tok_dataset):
        rng1 = random.Random(42)
        random_numbers = rng1.sample(range(0, len(tok_dataset)), size)
        tok_dataset = tok_dataset.select(list(random_numbers))
    
    return tok_dataset

#== Transformer Model Functions ===================================================================#
def create_tokenizer(model_path:str)->AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def create_peft_config(peft_strategy:str, model_path:str, N:int=None)->PeftConfig:    
    task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq(model_path) else TaskType.CAUSAL_LM

    if peft_strategy == 'prompt-tuning':
        # peft_config = PromptTuningConfig(
        #    task_type=task_type,
        #    prompt_tuning_init=PromptTuningInit.RANDOM,
        #    num_virtual_tokens=N,
        #    tokenizer_name_or_path=model_path
        # )
        peft_config = PromptTuningConfig(
           task_type=task_type,
           prompt_tuning_init=PromptTuningInit.TEXT,
           num_virtual_tokens=N,
           tokenizer_name_or_path=model_path,
           prompt_tuning_init_text="Select the option that best answers the following question",
        )

    elif peft_strategy == 'lora':
        peft_config = LoraConfig(
            task_type=task_type, inference_mode=False, r=N, lora_alpha=32, lora_dropout=0.1
        )

    elif peft_strategy == 'p-tuning':
        peft_config = PrefixTuningConfig(task_type=task_type, num_virtual_tokens=N)

    else:
        raise ValueError('invalid peft scheme')
    return peft_config

def create_base_model(model_path:str, dtype:str)->Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    D_TYPES = {'bfloat16':torch.bfloat16, 'float32':torch.float32}
    torch_dtype = D_TYPES[dtype]

    if model_path in CAUSAL_LLMS.values():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    elif model_path in SEQ2SEQ_LLMS.values():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        raise ValueError('invalid model')
    return model

def create_model(model_path:str, peft_config:str, dtype:str=None)->Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    model = create_base_model(model_path, dtype)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def load_peft_model(output_path:str, dtype:str):
    config = PeftConfig.from_pretrained(output_path)
    model = create_base_model(config.base_model_name_or_path, dtype)
    model = PeftModel.from_pretrained(model, output_path)
    return model

#== Optimizer Functions ===========================================================================#
def create_optimizer(model, lr, num_training_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.1*num_training_steps,
        num_training_steps=(num_training_steps),
    )
    return optimizer, lr_scheduler

#== Training / Validation Loops ===================================================================#
def get_class_logits(outputs:namedtuple, batch:dict, label_toks:list, device:str=None):
    # get class predictions
    mask = 1*(batch['labels'] != -100)
    label_tok_pos = mask.argmax(dim=1)
    pred_tok_pos = mask.argmax(dim=1)-1

    # get predicted token (from A, B, C, D)
    labels_tok = batch['labels'][torch.arange(outputs.logits.size(0)), label_tok_pos]
    labels = [(label_toks.index(tok) if tok in label_toks else None) for tok in labels_tok]
    if device:
        labels = torch.LongTensor(labels).to(device)

    # remove the output of the soft-prompts (if present)
    N = batch['input_ids'].size(1)
    outputs.logits = outputs.logits[:,-N:]
    
    # get class logit predictions
    vocab_logits = outputs.logits[torch.arange(outputs.logits.size(0)), pred_tok_pos]
    class_logits = vocab_logits[:, label_toks]
    return class_logits, labels

def train_loop(
    model, 
    train_dataloader, 
    loss_fn:str,
    optimizer, 
    lr_scheduler, 
    label_toks:list,
    device:str, 
    use_wandb:bool=False
)->Tuple[float, float]:
    model.train()
    loss_log = []
    hits_log = []
    train_pbar = tqdm(train_dataloader)
    for step, batch in enumerate(train_pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        # get model outputs
        outputs = model(**batch)
        
        # get class predictions and loss
        class_logits, labels_torch = get_class_logits(outputs, batch, label_toks, device)
        
        if loss_fn == 'ce':
            loss = F.cross_entropy(class_logits, labels_torch)
        elif loss_fn == 'll':
            loss = outputs.loss
            
        # do gradient updates
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        lr_scheduler.step()

        # log loss for tracking
        if not torch.isnan(loss).any().item():
            loss_log += [loss.cpu().detach().float()] 

        # log hits
        preds = class_logits.argmax(axis=-1).cpu().tolist()
        labels = labels_torch.cpu().tolist()

        hits = [i == j for i, j in zip(preds, labels)]
        hits_log.append(hits)
        
        if use_wandb:
            wandb.log({'loss':loss.item(), 'acc':np.mean(hits)})
        
        # update tqdm description
        train_pbar.set_description(f"loss={np.mean(loss_log[-50:]):.2f}  acc={100*np.mean(hits_log[-50:]):.1f}")

    avg_loss = np.mean(loss_log)
    acc = np.mean(hits_log)
    return avg_loss, acc

def eval_loop(model, dataloader, label_toks:list, device:str, use_wandb:bool=False)->float:
    model.eval()
    model.to(device)
    hits_log = []
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            outputs = model(**batch)
            
            pos = batch['attention_mask'].sum(dim=1) - 1
            N = batch['input_ids'].size(1)
            outputs.logits = outputs.logits[:,-N:]
    
            vocab_logits = outputs.logits[torch.arange(len(pos)), pos]
            class_logits = vocab_logits[:, label_toks]
            preds = class_logits.argmax(dim=-1)

            hits = (preds==labels).cpu().tolist()
            hits_log += hits
            pbar.set_description(f"acc: {100*np.mean(hits_log):.2f}  ")
                    
    if use_wandb:
        wandb.log({'eval-acc':np.mean(hits_log)})
        
    acc = np.mean(hits_log)
    return acc

#== Model Logging =================================================================================#
def setup_wandb(cfg: dict):
    #init wandb project
    wandb.init(
        project=f"peft-{cfg['dataset_name']}",
        entity='mg-speech-group',
        name=cfg['output_path'], 
        dir=cfg['output_path'],
    )

    wandb.config.update(cfg) 
        
#== Main function for running Parameter Efficient Finetuning ======================================#
def main(
    model_name:str,
    peft_strategy:str,
    dataset_name:str,
    output_path:str,
    loss_fn:str,
    num_epochs:int=3,
    batch_size:int=4,
    lr:int=1e-3,
    N:int=None,
    dtype:str=None, 
    train_size:int=None,
    val_size:int=None,
    max_length:int=512,
    device='cuda',
    use_wandb=False,
    no_save=False
):
    # set up tokenizer, model (with peft setup)
    model_path = MODEL_NAME_TO_PATH[model_name]
    tokenizer = create_tokenizer(model_path)
    peft_config = create_peft_config(peft_strategy, model_path, N)
    model = create_model(model_path, peft_config, dtype)

    # prepare dataset
    train_dataset, val_dataset, test_dataset = process_mcrc_dataset(dataset_name, model_path)
    
    # tokenize training datasets
    train_prc_fn = create_train_preprocess_function(tokenizer, max_length)
    tok_train_dataset = tokenize_dataset(train_dataset, train_prc_fn)
    tok_train_dataset = filter_dataset(tok_train_dataset, max_length, train_size)

    # tokenize validation/test datasets
    val_prc_fn = create_eval_preprocess_function(tokenizer, max_length=1024)
    tok_val_dataset = tokenize_dataset(val_dataset, val_prc_fn)
    tok_val_dataset = filter_dataset(tok_val_dataset, None, val_size)

    # tokenize validation/test datasets
    test_prc_fn = create_eval_preprocess_function(tokenizer, max_length=1024)
    tok_test_dataset = tokenize_dataset(test_dataset, test_prc_fn)
    tok_val_dataset = filter_dataset(tok_val_dataset, None, None)

    # create dataloaders
    train_dataloader = DataLoader(
        tok_train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    val_dataloader = DataLoader(
        tok_val_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        tok_test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    
    # create optimizer
    num_training_steps = len(train_dataloader) * num_epochs
    optimizer, lr_scheduler = create_optimizer(model, lr, num_training_steps)
    
    # determine label words
    label_words = ['A', 'B', 'C', 'D']
    label_toks = [int(tokenizer(word, add_special_tokens=False).input_ids[-1]) for word in label_words]

    # begin train loop
    model = model.to(device)
    best_acc = 0
    for epoch in range(num_epochs):
        trn_loss, trn_acc = train_loop(model, train_dataloader, loss_fn, optimizer, lr_scheduler, label_toks, device, use_wandb)
        val_acc = eval_loop(model, val_dataloader, label_toks, device, use_wandb)
        
        if val_acc > best_acc and not no_save:
            model.save_pretrained(output_path)

    # load model and run test
    if not no_save:
        model = load_peft_model(output_path, dtype)

    test_acc = eval_loop(model, test_dataloader, label_toks, device)
    print(f"final test acc {100*test_acc:.1f}")

    with open(f"{output_path}/test.txt", "w+") as file:
        file.write(f"{100*test_acc:.1f}")
        
def create_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help='which model to use as the base LLM')
    parser.add_argument('--dataset-name', type=str, required=True, help='which dataset to train on')
    parser.add_argument('--output-path', type=str, required=True, help='path to save model')

    parser.add_argument('--peft-strategy', type=str, required=True, help='which peft training scheme to use')
    parser.add_argument('--N', type=int, default=None, help='peft hyperparameter')
    parser.add_argument('--loss-fn', type=str, default='ce', help='which loss function to use')

    parser.add_argument('--batch-size', type=int, default=4, help='batchsize to use for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate to use for training')
    parser.add_argument('--num-epochs', type=int, default=None, help='number of epoechsto use for training')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='dtype to load model')
    parser.add_argument('--device', type=str, default='cuda', help='which device to use')
    parser.add_argument('--train-size', type=int, default=None, help='learning rate to use for training')
    parser.add_argument('--val-size', type=int, default=None, help='learning rate to use for training')
    
    parser.add_argument('--use-wandb', action='store_true', help='whether to track experiments with wandb')
    parser.add_argument('--no-save', action='store_true', help='whether not to save models')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    kwargs = vars(parser.parse_args())
    
    if kwargs['use_wandb']:
        setup_wandb(kwargs)
    
    main(**kwargs)