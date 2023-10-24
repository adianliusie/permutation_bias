import random
import re 

from datasets import load_dataset
from types import SimpleNamespace

from .load_mcrc import _create_splits

#== Hellaswag =====================================================================================#
def load_hellaswag():
    dataset = load_dataset("Rowan/hellaswag")
    train_data = list(dataset['train'])
    train, val = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    train, val, test = [format_hellaswag_split(split) for split in [train, val, test]]
    return train, val, test

def format_hellaswag_split(split_data):
    outputs = []
    for ex in split_data:
        ex_obj = SimpleNamespace(
            ex_id=ex['ind'], 
            question=None, 
            context=_hellaswag_pre_process(ex['ctx']), 
            options=[_hellaswag_pre_process(x) for x in ex['endings']], 
            label=int(ex['label'])
        )
        outputs.append(ex_obj)
    return outputs

def _hellaswag_pre_process(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

#== ARC ===========================================================================================#
def load_arc(level):
    if level=='easy':
        dataset = load_dataset('ai2_arc', 'ARC-Easy') 
    if level=='challenge':
        dataset = load_dataset('ai2_arc', 'ARC-Challenge') 
    
    train = list(dataset['train'])
    val = list(dataset['validation'])
    test = list(dataset['test'])

    train, val, test = [format_arc_split(split) for split in [train, val, test]]
    return train, val, test

def format_arc_split(split_data):
    outputs = []
    for ex in split_data:
        ex_obj = SimpleNamespace(
            ex_id=ex['id'], 
            question=ex['question'], 
            context=None, 
            options=ex['choices']['text'], 
            label=ex['choices']['label'].index(ex['answerKey'])
        )
        if len(ex_obj.options) == 4:
            outputs.append(ex_obj)
    return outputs

#== MedMCQA =======================================================================================#
def load_medmcqa():
    dataset = load_dataset('medmcqa') 

    train_data = list(dataset['train'])
    train, val = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    train, val, test = [format_medmcqd_split(split) for split in [train, val, test]]
    return train, val, test

def format_medmcqd_split(split_data):
    outputs = []
    for ex in split_data:
        ex_obj = SimpleNamespace(
            ex_id=ex['id'], 
            question=ex['question'], 
            context=None, 
            options=[ex['opa'], ex['opb'], ex['opc'], ex['opd']], 
            label=int(ex['cop'])
        )
        outputs.append(ex_obj)
    return outputs
