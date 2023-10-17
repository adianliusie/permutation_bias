import itertools
from scipy.special import softmax
from typing import Dict
import numpy as np

from .general import load_pickle
from ..models.llama import PromptedLlama2System
from ..models.flant5 import PromptedFlanT5System
from ..data import load_mcrc_data

#== load data ==============================================================#
def load_perm_data(system_name:str, dataset:str, num_prompts:int=3):
    data = [load_pickle(f'outputs/{dataset}/{system_name}_p{p_num}.pk') for p_num in range(num_prompts)]
    avg = get_prompt_ensemble_data(data)
    return data, avg

def get_prompt_ensemble_data(data):
    avg = []
    for i in range(len(data[0])):
        temp_dict = {}
        for key in data[0][0].keys():
            if key == 'q_id':
                temp_dict['q_id'] = data[0][i]['q_id']
            else:
                temp_dict[key] = np.mean([data[k][i][key] for k in range(len(data))], axis=0)
        avg.append(temp_dict)
    return avg

#== getting attributes =========-==========================================#
def probs_matrix(x:Dict[str, np.ndarray]):
    return softmax(x['logits'], axis=-1)

def pred_matrix(x:dict):
    max_logits = x['logits'].max(axis=1)[..., np.newaxis]
    return 1*(x['logits'] == max_logits)
    
def pred_list(x:dict):
    return np.argmax(x['logits'], axis=1)

#== Priors ===============================================================#
def hard_prior(x:dict):
    return pred_matrix(x).mean(axis=0)

def soft_prior(x:dict):
    return probs_matrix(x).mean(axis=0)

#== Prior_matching method ================================================#
def prior_matching_logits(x:dict):
    logits = x['logits']
    N = logits.shape[-1]
    uniform = (1/N)*np.ones(N)
    weights = np.zeros(N)
    perm_prior = softmax(logits, axis=-1).mean(axis=0)

    while np.abs(perm_prior-uniform).sum() > 0.005:
        update = 1*(perm_prior < uniform)
        weights += 0.01*update
        logits = x['logits'] + weights[np.newaxis, ...]
        perm_prior = softmax(logits, axis=-1).mean(axis=0)
    
    return logits

def prior_match_pred_list(x:dict):
    logits = prior_matching_logits(x)
    max_logits = logits.max(axis=1)[..., np.newaxis]
    pred_matrix = 1*(logits == max_logits)
    pred_list = np.argmax(pred_matrix, axis=1)
    return pred_list

#== Ensembling ===========================================================#
def soft_ensemble(x):
    probs = probs_matrix(x)
    perms = np.array(list(itertools.permutations([0,1,2,3])))
    back_perms = np.argsort(perms)

    reverted_probs = probs[np.arange(0,24)[..., np.newaxis], back_perms]
    return reverted_probs.mean(axis=0)
 
def max_voter_ensemble(x):
    preds = pred_matrix(x)
    perms = np.array(list(itertools.permutations([0,1,2,3])))
    back_perms = np.argsort(perms)

    reverted_probs = preds[np.arange(0,24)[..., np.newaxis], back_perms]
    return reverted_probs.mean(axis=0)

#== Calculating accuracy ================================================#
def base_accuracy(x:dict):
    preds = pred_list(x)
    return int(preds[0] == x['labels'][0])

def all_perm_accuracy(x:dict):
    preds = pred_list(x)
    return (preds == x['labels']).mean()

def prior_match_accuracy(x):
    preds = prior_match_pred_list(x)
    return int(preds[0] == x['labels'][0])

def ensemble_accuracy(x:dict):
    pred = soft_ensemble(x).argmax(axis=0)
    return int(pred == x['labels'][0])

def max_voter_accuracy(x:dict):
    pred = max_voter_ensemble(x).argmax(axis=0)
    return int(pred == x['labels'][0])

#== Null Input methods ================================================#
def get_prompt_template(system_name:str):
    prompt_templates = [
        '{context}\n\n{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}',
        'Context: {context}\n\nQuestion:{question}\nA:{option_1}\nB:{option_2}\nC:{option_3}\nD:{option_4}',
        'Content{context}\n\nQuestion:{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}\n\nWhich option best answers the question: A, B, C or D?'
    ]

    if 'llama' in system_name:
        prompt_templates = [template + '\n\nAnswer:' for template in prompt_templates]
        prompt_templates[2] = prompt_templates[2].replace('Answer', 'Decision')

    return prompt_templates

label_words = ['A', 'B', 'C', 'D']

def load_system(system_name, device):
    if 'flan' in system_name:
        system = PromptedFlanT5System(
            system_name=system_name,
            decoder_prefix='',
            label_words=[' '+x for x in label_words],
            device=device
        )

    elif ('llama' in system_name) or ('vicuna' in system_name):
        system = PromptedLlama2System(
            system_name=system_name,
            decoder_prefix='',
            label_words=label_words,
            device=device
        )
    return system

SYSTEM=None
SYSTEM_NAME=None
DATA_NAME=None
MCRC_DATA=None
def null_norm_logits(x:dict, p_num:int, system_name, dataset:list, device='cuda'):
    #==load model and data if not already loaded ======
    global SYSTEM
    global SYSTEM_NAME
    global MCRC_DATA
    global DATA_NAME

    if SYSTEM_NAME != system_name:
        SYSTEM_NAME = system_name
        SYSTEM = load_system(system_name, device=device)
    
    if DATA_NAME != dataset:
        DATA_NAME = dataset
        _, _, MCRC_DATA = load_mcrc_data(dataset)
    
    #== prepare null input ============================
    prompt_templates = get_prompt_template(system_name)
    prompt_template = prompt_templates[p_num]
    
    q_data = [ex for ex in MCRC_DATA if ex.ex_id == x['q_id']][0]

    null_template = prompt_template.format(
        context=q_data.context,
        question=q_data.question,
        option_1='option 1',
        option_2='option 2',
        option_3='option 3',
        option_4='option 4'
    )
    
    #== get output ============================
    output = SYSTEM.prompt_classifier_response(input_text=null_template)
    logits = np.array(output.logits)
    new_logits = x['logits'] -  logits[np.newaxis, ...]
    return new_logits

def null_norm_pred_list(x:dict, p_num:int, system_name, dataset:list, device='cuda'):
    logits = null_norm_logits(x, p_num, system_name, dataset, device=device)
    max_logits = logits.max(axis=1)[..., np.newaxis]
    pred_matrix = 1*(logits == max_logits)
    pred_list = np.argmax(pred_matrix, axis=1)
    return pred_list

def null_norm_accuracy(x:dict, p_num:int, system_name, dataset:list, device='cuda'):
    preds = null_norm_pred_list(x, p_num, system_name, dataset, device=device)
    return int(preds[0] == x['labels'][0])