import os
import csv
import random
import pandas as pd
import numpy as np

from copy import deepcopy
from typing import List, Tuple
from datasets import load_dataset
from types import SimpleNamespace

from ..utils.general import save_pickle, load_pickle, load_json, get_base_dir
from .download import RACE_C_DIR, download_race_plus_plus
from .download import RECLOR_DIR, download_reclor

#== main data loading methods ==================================================================#
def load_race(levels=['M', 'H', 'C']):
    #load RACE-M and RACE-H data from hugginface
    race_data = {}
    if 'M' in levels: race_data['M'] = load_dataset("race", "middle")
    if 'H' in levels: race_data['H'] = load_dataset("race", "high")
    if 'C' in levels: race_data['C'] = load_race_c()

    #load and format each split, for each difficulty level, and add to data
    SPLITS = ['train', 'validation', 'test']
    train_all, dev_all, test_all = [], [], []
    for key, data in race_data.items():
        train, dev, test = [format_race(data[split], key) for split in SPLITS]
        train_all += train
        dev_all   += dev
        test_all  += test

    return train_all, dev_all, test_all

def load_cosmos():
    #load RACE-M and RACE-H data from hugginface
    dataset = load_dataset("cosmos_qa")

    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])

    train = format_cosmos(train)
    dev = format_cosmos(dev)
    test = format_cosmos(test)
    
    return train, dev, test

def load_reclor():    
    # download data if missing
    if not os.path.isdir(RECLOR_DIR):
        download_reclor()
    
    # load and prepare each data split
    splits_path = [f'{RECLOR_DIR}/{split}.json' for split in ['train', 'val', 'val']]
    train, dev, test = [load_reclor_split(path) for path in splits_path]
    return train, dev, test

def load_camchoice():
    BASE_DIR = get_base_dir()
    data_path = f'{BASE_DIR}/data/CUPA_MC4.csv'
    
    #load data and convert from weird windows format
    df = pd.read_csv(data_path, encoding='cp1252')
    df = df.replace({'’': "'"}, regex=True)
    df = df.replace({'‘': "'"}, regex=True)
    df = df.replace({'“': '"'}, regex=True)
    df = df.replace({'”': '"'}, regex=True)
    df = df.replace({'–':'-'}, regex=True)
    data = format_camchoice_csv(df)
    return None, None, data

#== General loading utils ==========================================================================# 
def format_race(data, char):
    """ converts dict to SimpleNamespace for QA data"""
    outputs = []
    ans_to_id = {'A':0, 'B':1, 'C':2, 'D':3}
    for k, ex in enumerate(data):
        ex_obj = SimpleNamespace(
            ex_id=f"{char}_{k}", 
            question=ex['question'], 
            context=ex['article'], 
            options=ex['options'], 
            label=ans_to_id[ex['answer']]
        )
        outputs.append(ex_obj)
    return outputs

def format_cosmos(data:List[dict]):
    outputs = []
    for ex in data:
        ex_obj = SimpleNamespace(
            ex_id=ex['id'], 
            question=ex['question'], 
            context=ex['context'], 
            options=[ex[f'answer{k}'] for k in [0,1,2,3]], 
            label=ex['label']
        )
        outputs.append(ex_obj)
    return outputs

def load_reclor_split(split_path:str):
    split_data = load_json(split_path)
    outputs = []
    for ex in split_data:
        ex_obj = SimpleNamespace(
            ex_id=ex['id_string'], 
            question=ex['question'], 
            context=ex['context'], 
            options=ex['answers'], 
            label=ex['label']
        )
        outputs.append(ex_obj)
    return outputs

def get_camchoice_q_headers(df):
    """ gets all column headers for the question"""
    questions = [x.replace('A','') for x in df.columns if x[0]=='Q' and x[-1]=='A']
    return questions

def format_camchoice_csv(df:pd.DataFrame)->List[SimpleNamespace]:
    ans_to_id = {'A':0, 'B':1, 'C':2, 'D':3}
    q_headers = get_camchoice_q_headers(df)

    output = []
    for index, row in df.iterrows():
        context = row['Text']
        q_headers = [q for q in q_headers if not pd.isna(row[q])]
        for q_num in q_headers:
            #get target level
            target_level = row['Target Level'].lower()

            # Get specific question details
            ex_id = f'{target_level}_{index}_{q_num}'
            question = row[q_num]
            options  = [row[f'{q_num}{i}'] for i in ['A', 'B', 'C', 'D']]
            answer   = row[f'{q_num}_answer']
            answer   = ans_to_id[answer]

            # Get candidates chosen answer distribution if valid
            #cand_dist   = [row[f'{q_num}_distract_{i}_fac'] for i in ['a', 'b', 'c', 'd']]
            #disc_scores = [row[f'{q_num}_distract_{i}_disc'] for i in ['a', 'b', 'c', 'd']] 
            #cand_dist   = process_cand_dist(cand_dist, disc_scores)

            #process output type
            ex_obj = SimpleNamespace(
                         ex_id=ex_id, 
                         question=question, 
                         context=context, 
                         options=options, 
                         label=answer,
                     )
            output.append(ex_obj)     
    return output

def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

#== Loading for RACE-C ============================================================================#
def load_race_c():    
    # Download data if missing
    if not os.path.isdir(RACE_C_DIR):
        download_race_plus_plus()
    
    # Load cached data if exists, otherwise process and cache
    pickle_path = os.path.join(RACE_C_DIR, 'cache.pkl')    
    if os.path.isfile(pickle_path):
        train, dev, test = load_pickle(pickle_path)
    else:
        splits_path = [f'{RACE_C_DIR}/{split}' for split in ['train', 'dev', 'test']]
        train, dev, test = [load_race_c_split(path) for path in splits_path]
        save_pickle(data=[train, dev, test], path=pickle_path)
        
    return {'train':train, 'validation':dev, 'test':test}

def load_race_c_split(split_path:str):
    file_paths = [f'{split_path}/{f_path}' for f_path in os.listdir(split_path)]
    outputs = []
    for file_path in file_paths:
        outputs += load_race_file(file_path)
    return outputs

def load_race_file(path:str):
    file_data = load_json(path)
    article = file_data['article']
    answers = file_data['answers']
    options = file_data['options']
    questions = file_data['questions']
    
    outputs = []
    assert len(questions) == len(options) == len(answers)
    for k in range(len(questions)):
        ex = {'question':questions[k], 
              'article':article,
              'options':options[k],
              'answer':answers[k]}
        outputs.append(ex)
    return outputs
