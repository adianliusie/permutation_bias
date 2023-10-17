import random

from datasets import load_dataset
from types import SimpleNamespace

from .load_mcrc import _create_splits

def load_hellaswag():
    dataset = load_dataset("Rowan/hellaswag")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    #for now just sample 2000 test samples
    #random.shuffle(test)
    #test = test[:2000]
    
    train, dev, test = [format_split(split) for split in [train, dev, test]]
    return train, dev, test

def format_split(split_data):
    outputs = []
    for ex in split_data:
        ex_obj = SimpleNamespace(
            ex_id=ex['ind'], 
            question=None, 
            context=ex['ctx'], 
            options=ex['endings'], 
            label=int(ex['label'])
        )
        outputs.append(ex_obj)
    return outputs
