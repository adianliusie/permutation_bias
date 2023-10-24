import itertools
import random
import math

from copy import deepcopy
from typing import List
from types import SimpleNamespace

from .load_nlg import load_summeval, load_podcast
from .load_mcrc import load_race, load_reclor, load_cosmos, load_camchoice
from .load_mcqa import load_hellaswag, load_arc, load_medmcqa

def get_dataset_category(dataname):
    if dataname.endswith('-s'):
        dataname = dataname[:-2]
    
    if dataname in [
        'race++', 'race', 'cosmos', 'reclor', 'hellaswag', 'arc-easy', 'arc-challenge', 'medmcqa'
    ]:
        return 'multiple_choice'
    if dataname in ['summeval']:
        return 'comparative'

def load_nlg_data(dataname):
    if dataname == 'summeval': data = load_summeval()
    elif dataname == 'podcast': data = load_podcast()
    return data

def load_mcqa_data(dataname, all_perm=False, lim=None):
    if dataname.endswith('-s'):
        lim = 200
        dataname = dataname[:-2]
    
    if   dataname == 'race++': train, dev, test = load_race(levels=['M', 'H', 'C'])
    elif dataname == 'race':   train, dev, test = load_race(levels=['M', 'H'])
    elif dataname == 'race-M': train, dev, test = load_race(levels=['M'])
    elif dataname == 'race-H': train, dev, test = load_race(levels=['H'])
    elif dataname == 'race-C': train, dev, test = load_race(levels=['C'])
    elif dataname == 'reclor': train, dev, test = load_reclor()
    elif dataname == 'cosmos': train, dev, test = load_cosmos()
    elif dataname == 'camchoice': train, dev, test = load_camchoice()
    elif dataname == 'hellaswag': train, dev, test = load_hellaswag()
    elif dataname == 'arc-easy': train, dev, test = load_arc(level='easy')
    elif dataname == 'arc-challenge': train, dev, test = load_arc(level='challenge')
    elif dataname == 'medmcqa': train, dev, test = load_medmcqa()
    else: raise ValueError(f"{dataname}: invalid dataset name") 

    if lim:
        train = rand_select(train, lim)
        dev   = rand_select(dev, lim)
        test  = rand_select(test, lim)
    
    if all_perm:
        train, dev, test = [all_permutations(split) for split in [train, dev, test]]

    return train, dev, test    

#== Misc utils functions ==========================================================================#
def all_permutations(data:List[SimpleNamespace]):
    if data is None: 
        return None
    data = deepcopy(data)
    output = []
    for ex in data:
        N_questions = len(ex.options)
        indice_permutations = itertools.permutations(list(range(N_questions)))
        for k, indices in enumerate(indice_permutations):
            ex_copy = deepcopy(ex)
            ex_copy.options = [ex.options[k] for k in indices]
            ex_copy.label = indices.index(ex.label)
            ex_copy.ex_id = f"{ex.ex_id}-p{k}"

            if N_questions<5:
                output.append(ex_copy)
            elif random.random()>(50/math.factorial(N_questions)):
                output.append(ex_copy)

    return output

def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]