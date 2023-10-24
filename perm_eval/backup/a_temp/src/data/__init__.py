import itertools

from copy import deepcopy
from typing import List
from types import SimpleNamespace

from .load_nlg import load_summeval, load_podcast
from .load_mcrc import load_race, load_reclor, load_cosmos, load_camchoice
from .load_mc_ending import load_hellaswag


def get_dataset_category(data_name):
    if data_name in ['race++', 'race', 'cosmos', 'camchoice', 'hellaswag']:
        return 'multiple_choice'
    if data_name in ['summeval']:
        return 'comparative'

def load_nlg_data(data_name):
    if data_name == 'summeval': data = load_summeval()
    elif data_name == 'podcast': data = load_podcast()
    return data

def load_mcrc_data(data_name, all_perm=False, lim=None):
    if   data_name == 'race++': train, dev, test = load_race(levels=['M', 'H', 'C'])
    elif data_name == 'race':   train, dev, test = load_race(levels=['M', 'H'])
    elif data_name == 'race-M': train, dev, test = load_race(levels=['M'])
    elif data_name == 'race-H': train, dev, test = load_race(levels=['H'])
    elif data_name == 'race-C': train, dev, test = load_race(levels=['C'])
    elif data_name == 'reclor': train, dev, test = load_reclor()
    elif data_name == 'cosmos': train, dev, test = load_cosmos()
    elif data_name == 'camchoice': train, dev, test = load_camchoice()
    elif data_name == 'hellaswag': train, dev, test = load_hellaswag()
    else: raise ValueError(f"{data_name}: invalid dataset name") 
    
    if lim:
        train = rand_select(train, lim)
        dev   = rand_select(dev, lim)
        test  = rand_select(test, lim)
    
    if all_perm:
        train, dev, test = [all_permutations(split) for split in [train, dev, test]]

    return train, dev, test    

def all_permutations(data:List[SimpleNamespace]):
    if data is None: 
        return None
    data = deepcopy(data)
    output = []
    for ex in data:
        indice_permutations = itertools.permutations([0,1,2,3])
        for k, indices in enumerate(indice_permutations):
            ex_copy = deepcopy(ex)
            ex_copy.options = [ex.options[k] for k in indices]
            ex_copy.label = indices.index(ex.label)
            ex_copy.ex_id = f"{ex.ex_id}-p{k}"
            output.append(ex_copy)
    return output
