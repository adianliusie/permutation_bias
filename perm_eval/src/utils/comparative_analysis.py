import numpy as np
import itertools
from scipy.special import softmax
from typing import Dict
import numpy as np

from .general import load_pickle
from ..models.llama import PromptedLlama2System
from ..models.flant5 import PromptedFlanT5System
from ..data import load_nlg_data
from ..models.llama import PromptedLlama2System
from ..models.flant5 import PromptedFlanT5System
from ..data import load_nlg_data

#== load data ==============================================================#
def load_comp_data(system_name:str, dataset:str, score_type:str):
    data = [load_pickle(f'outputs/{dataset}/{score_type}_{system_name}_p{p_num}.pk') for p_num in range(3)]
    avg = get_prompt_ensemble_data(data)
    return data, avg

def get_prompt_ensemble_data(data):
    avg = []
    for i in range(len(data[0])):
        temp_dict = {}
        for key in data[0][0].keys():
            if key == 'c_num':
                temp_dict['c_num'] = data[0][i]['c_num']
            else:
                temp_dict[key] = np.mean([data[k][i][key] for k in range(len(data))], axis=0)
        avg.append(temp_dict)
    return avg

#== getting attributes =========-==========================================#
def N(x):
    return x['logits'].shape[0]

def probs_matrix(x:Dict[str, np.ndarray]):
    output = softmax(x['logits'], axis=-1)
    output[np.eye(N(x)) == 1] = 0
    return output

def one_hot_pred_matrix(x:dict):
    max_logits = x['logits'].max(axis=-1)[..., np.newaxis]
    output = 1*(x['logits'] == max_logits)
    output[np.eye(N(x)) == 1] = 0
    return output

def pred_matrix(x:dict):
    return np.argmax(one_hot_pred_matrix(x), axis=-1)

#== Priors ===============================================================#
def hard_prior(x:dict):   
    print('weird function- maybe ignore this result')
    prior_num = one_hot_pred_matrix(x).sum(axis=0).sum(axis=0)
    prior_denom = N(x)*(N(x)-1)
    return prior_num/prior_denom

def soft_prior(x:dict):
    prior_num = probs_matrix(x).sum(axis=0).sum(axis=0)
    prior_denom = N(x)*(N(x)-1)
    return prior_num/prior_denom

#== Prior_matching method ================================================#
def prior_matching_logits(x:dict):
    logits = x['logits']
    N = logits.shape[-1]
    uniform = (1/N)*np.ones(N)
    weights = np.zeros(N)
    perm_prior = softmax(logits, axis=-1).mean(axis=0).mean(axis=0)
    
    while np.abs(perm_prior-uniform).sum() > 0.005:
        update = 1*(perm_prior < uniform)
        weights += 0.01*update
        logits = x['logits'] + weights[np.newaxis, np.newaxis, ...]
        perm_prior = softmax(logits, axis=-1).mean(axis=0).mean(axis=0)
    
    return logits

def prior_match_pred_matrix(x:dict):
    return np.argmax(prior_matching_logits(x), axis=-1)

#== Ensembling ===========================================================#
def soft_ensemble(x):
    probs = probs_matrix(x)
    avg_probs = (probs + 1-np.transpose(probs, (1, 0, 2)))/2
    return avg_probs
 
def max_voter_ensemble(x):
    preds = pred_matrix(x)
    avg_preds = (preds + 1-preds.T)/2
    preds = 1*(avg_preds > 0)
    return preds

#== Calculating hits/accuracy ============================================#
def _get_hits(x, preds):
    scores = np.array(x['labels'])
    labels = 1*(scores[..., np.newaxis] < scores[np.newaxis, ...])
    mask = (scores[..., np.newaxis] != scores[np.newaxis, ...])
    hits = (preds==labels)[mask]
    return hits

def base_hits(x:dict):
    preds = pred_matrix(x)
    return _get_hits(x, preds)

def prior_match_hits(x):
    preds = prior_match_pred_matrix(x)
    return _get_hits(x, preds)

def ensemble_hits(x:dict):
    preds = np.argmax(soft_ensemble(x), axis=-1) 
    return _get_hits(x, preds)

#== Calculating hits/accuracy ============================================#
def _calc_accuracy(x, preds):
    hits = _get_hits(x, preds)
    return hits.mean()

def base_accuracy(x:dict):
    preds = pred_matrix(x)
    return _calc_accuracy(x, preds)

def prior_match_accuracy(x):
    preds = prior_match_pred_matrix(x)
    return _calc_accuracy(x, preds)

def ensemble_accuracy(x:dict):
    preds = np.argmax(soft_ensemble(x), axis=-1) 
    return _calc_accuracy(x, preds)

def max_voter_accuracy(x:dict):
    preds = max_voter_ensemble(x)
    return _calc_accuracy(x, preds)

#== Load hardcoded prompts and label words =================================================================#
def get_prompt_template(system_name:str):
    prompt_templates = [
        '{context}\n\nWhich Summary is more {adjective}?\n\nSummary A: {response_1}\nSummary B: {response_2}',
        '{context}\n\nSummary A: {response_1}\nSummary B: {response_2}\n\nWhich Summary is more {adjective} with respect to the article, Summary A or Summary B?',
        'Assess the following two summaries given the corresponding passage, and determine which summary is more {adjective}.\n\nPassage:\n{context}\n\nSummary A: {response_1}\nSummary B: {response_2}\n\nWhich Summary is more {adjective}?',
    ]

    if 'llama' in system_name:
        prompt_templates = [template + '\n\nAnswer:' for template in prompt_templates]
        prompt_templates[2] = prompt_templates[2].replace('Answer', 'Decision')

    return prompt_templates

label_words = ['A', 'B']

#== Syatem Loader method ====================================================================================#
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

#== Main null norm function ==================================================================================#
SYSTEM=None
SYSTEM_NAME=None
DATA_NAME=None
NLG_DATA=None

def null_norm_logits(x:dict, p_num:int, system_name, dataset:str, score_type:str, device='cuda'):
    #==load model and data if not already loaded ======
    global SYSTEM
    global SYSTEM_NAME
    global NLG_DATA
    global DATA_NAME

    if SYSTEM_NAME != system_name:
        SYSTEM_NAME = system_name
        SYSTEM = load_system(system_name, device=device)
    
    if DATA_NAME != dataset:
        DATA_NAME = dataset
        NLG_DATA = load_nlg_data(dataset)
    
    #== prepare null input ============================
    prompt_templates = get_prompt_template(system_name)
    prompt_template = prompt_templates[p_num]
    
    c_data = [ex for ex in NLG_DATA if ex.context_id == x['c_num']][0]

    score_to_adjective = {
        'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant', 
        'naturalness':'natural', 'continuity':'good a continuation', 'engagingness':'engaging', 
        'grammar':'grammatically correct', 'overall':'good', 'semantic':None, 'good': 'good'
    }
    adjective = score_to_adjective[score_type]
    null_template = prompt_template.format(
        context=c_data.context, 
        response_1='',
        response_2='',
        adjective=adjective
    )
    
    #== get output ============================
    output = SYSTEM.prompt_classifier_response(input_text=null_template)
    logits = np.array(output.logits)
    new_logits = x['logits'] -  logits[np.newaxis, np.newaxis, ...]
    return new_logits

def null_norm_pred_matrix(x:dict, p_num:int, system_name:str, dataset:str, score_type:str, device='cuda'):
    logits = null_norm_logits(x, p_num, system_name, dataset, score_type, device=device)
    pred_matrix = np.argmax(logits, axis=-1)
    return pred_matrix

def null_norm_hits(x, *args, **kwargs):
    preds = null_norm_pred_matrix(x, *args, **kwargs)
    return _get_hits(x, preds)

def null_norm_accuracy(x, *args, **kwargs):
    preds = null_norm_pred_matrix(x, *args, **kwargs)
    return _calc_accuracy(x, preds)
