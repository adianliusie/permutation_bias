import argparse
import torch
import numpy as np
import torch.nn.functional as F
import wandb
import random
import random 
import itertools

from copy import deepcopy
from collections import defaultdict


from tqdm import tqdm
from typing import Union, Callable, Tuple
from collections import namedtuple
from src.data import load_mcrc_data

from peft_finetune import MODEL_NAME_TO_PATH
from peft_finetune import is_seq2seq, create_tokenizer, create_peft_config, create_model, load_peft_model, create_optimizer

#== DataLoader Class for tokenization and batching ================================================#
class SelfConsistencyDataLoader:
    def __init__(self, data, tokenizer, bsz:int, shuffle:bool=False):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bsz = bsz
        self.shuffle = shuffle
        self.rng = random.Random(42)

    def create_batches(self):
        PERMUTATIONS = list(itertools.permutations([0,1,2,3]))
        
        data = deepcopy(self.data)
        if self.shuffle:
            random.shuffle(data)

        for ex_perms in data:
            inds = self.rng.sample(range(0, len(ex_perms)), self.bsz)
            ex_list = [ex_perms[i] for i in inds]
            perm_list = torch.LongTensor([PERMUTATIONS[i] for i in inds])
            
            inputs = self.tokenizer([ex['text'] for ex in ex_list], padding=True, return_tensors="pt")
            labels = [ex['label_id'] for ex in ex_list]
            
            batch = {
                'input_ids':inputs['input_ids'],
                'attention_mask':inputs['attention_mask'],
                'permutations':perm_list,
                'labels':labels
            }
            yield batch

    def __iter__(self):
        self.batches = self.create_batches()
        return self

    def __next__(self):
        return next(self.batches)

    def __len__(self):
        return len(self.data)
    
#== data creation methods =========================================================================#
def create_allperm_data(dataset_name:str, model_name:str, max_len:int=None):
    LABEL_WORDS = ['A', 'B', 'C', 'D']
    prompt_template = '{context}\n{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}'
    
    if not is_seq2seq(model_name):
        prompt_template = prompt_template + '\n\nAnswer:'

    # load data
    train, val, test = load_mcrc_data(dataset_name, all_perm=True)

    train_data, val_data, test_data = [], [], []
    for split, output in [(train, train_data), (val, val_data), (test, test_data)]:
        grouped_data = defaultdict(list)
        for ex in split:
            grouped_data[ex.ex_id.split('-')[0]] += [ex]

        for ex_perms in grouped_data.values():
            context_out = []
            for ex in ex_perms:
                input_text = prompt_template.format(
                    context=ex.context, 
                    question=ex.question,
                    option_1=ex.options[0],
                    option_2=ex.options[1],
                    option_3=ex.options[2],
                    option_4=ex.options[3]
                )
                label_text = LABEL_WORDS[ex.label]
                context_out.append(({'text':input_text, 'label_text':label_text, 'label_id':ex.label}))
            output.append(context_out)
            
    return train_data, val_data, test_data

def filter_data(data, size:int=None):
    if size:
        new_data = deepcopy(data)
        random.shuffle(new_data)
        new_data = new_data[:size]
    return new_data
    
#== Training / Validation Loops ===================================================================#
def train_loop(
    model, 
    train_dataloader, 
    loss_fn:str,
    optimizer, 
    lr_scheduler, 
    label_toks:list,
    device:str, 
    use_wandb:bool=False,
    eval=False
)->Tuple[float, float]:
    model.train()
    loss_log = []
    hits_log = []

    train_pbar = tqdm(train_dataloader)
    for step, batch in enumerate(train_pbar):
        labels = batch.pop('labels')
        batch = {k: v.to(device) for k, v in batch.items()}
        perms = batch.pop('permutations')

        # get model outputs
        outputs = model(**batch)
        
        vocab_logits = outputs.logits[:, -1]
        class_logits = vocab_logits[:, label_toks]
        probs = F.softmax(class_logits, dim=-1)

        back_perms = torch.argsort(perms)
        ordered_p = torch.gather(probs, 1, back_perms)

        log_p = torch.log(ordered_p)
        loss = -1*torch.mean(ordered_p.unsqueeze(0)*log_p.unsqueeze(1))
        
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
        hits = [i == j for i, j in zip(preds, labels)]
        hits_log.append(hits)
        
        if use_wandb and not eval:
            wandb.log({'loss':loss.item(), 'acc':np.mean(hits)})
        
        # update tqdm description
        train_pbar.set_description(f"loss={np.mean(loss_log[-50:]):.2f}  acc={100*np.mean(hits_log[-50:]):.1f}")

    avg_loss = np.mean(loss_log)
    acc = np.mean(hits_log)

    if use_wandb and eval:
        wandb.log({'eval-loss':avg_loss, 'eval-acc':acc})
    
    return avg_loss, acc

#== Model Logging =================================================================================#
def setup_wandb(cfg: dict):
    #init wandb project
    wandb.init(
        project=f"peft2-{cfg['dataset_name']}",
        entity='mg-speech-group',
        name=cfg['output_path'].replace('models/', ''), 
        dir=cfg['output_path']
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
    train_dataset, val_dataset, test_dataset = create_allperm_data(dataset_name, model_path)
    
    train_dataset = filter_data(train_dataset, train_size)
    val_dataset = filter_data(val_dataset, val_size)

    # tokenize training datasets
    train_dataloader = SelfConsistencyDataLoader(train_dataset, tokenizer, bsz=batch_size, shuffle=True)
    val_dataloader = SelfConsistencyDataLoader(val_dataset, tokenizer, bsz=batch_size, shuffle=False)
    
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
        val_loss, val_acc = train_loop(model, val_dataloader, loss_fn, optimizer, lr_scheduler, label_toks, device, use_wandb, eval=True)
        
        if val_acc > best_acc and not no_save:
            model.save_pretrained(output_path)

    # load model and run test
    #if not no_save:
    #    model = load_peft_model(output_path, dtype)

    # test_acc = eval_loop(model, test_dataloader, label_toks, device)
    # print(f"final test acc {100*test_acc:.1f}")

    # with open(f"{output_path}/test.txt", "w+") as file:
    #     file.write(f"{100*test_acc:.1f}")
        
def create_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help='which model to use as the base LLM')
    parser.add_argument('--dataset-name', type=str, required=True, help='which dataset to train on')
    parser.add_argument('--output-path', type=str, required=True, help='path to save model')

    parser.add_argument('--peft-strategy', type=str, required=True, help='which peft training scheme to use')
    parser.add_argument('--N', type=int, default=None, help='peft hyperparameter')
    parser.add_argument('--loss-fn', type=str, default=None, help='which loss function to use')

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