import os
import string
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict

from src.data import get_dataset_category, load_mcqa_data, load_nlg_data
from src.models.llama import PromptedLlama2System
from src.models.flant5 import PromptedFlanT5System
from src.utils.general import save_pickle

from src.prompts import get_prompt_template, get_label_words, SCORE_TO_ADJECTIVE

def run_bias_search(
    dataset:str,
    system_name:str,
    score_type:str=None,
    peft_path:str=None,
    device:str='cuda'
):
    #== define prompt templates for task ==========================================================#
    prompt_templates = get_prompt_template(dataset)
    decoder_prefix, label_words = get_label_words(dataset)
    dataset_category = get_dataset_category(dataset)

    #== data depending on task ====================================================================#
    if dataset_category == 'multiple_choice':
        _, _, data = load_mcqa_data(dataset, all_perm=True)
        grouped_data = defaultdict(list)
        for ex in data:
            grouped_data[ex.ex_id.split('-')[0]] += [ex]
        data = [v for v in grouped_data.values()]

    elif dataset_category == 'comparative':
        data = load_nlg_data(dataset)
        adjective = SCORE_TO_ADJECTIVE[score_type]

    #== load LLM system ===========================================================================#
    if 'flan' in system_name:
        system = PromptedFlanT5System(
            system_name=system_name,
            decoder_prefix=decoder_prefix,
            label_words=[' '+x for x in label_words],
            device=device
        )

    elif ('llama' in system_name) or ('vicuna' in system_name):
        system = PromptedLlama2System(
            system_name=system_name,
            decoder_prefix=decoder_prefix,
            label_words=label_words,
            device=device
        )
    
    if peft_path:
        system.load_peft_model(peft_path)

    #== Make Dir if not already exists ============================================================#
    if not os.path.isdir(f"outputs/{dataset}"):
        os.makedirs(f"outputs/{dataset}")
    
    #== Go through all selected examples ==========================================================#
    for p_num, prompt_template in enumerate(prompt_templates):
        if not peft_path:
            output_path = f"outputs/{dataset}/{system_name}_p{p_num}.pk"
        else:
            peft_str = peft_path.replace('peft_models/', '').replace('/', '-')
            output_path = f"outputs/{dataset}/{peft_str}_p{p_num}.pk"

        #skip files already processed
        if os.path.isfile(output_path):
            continue

        #== code for multiplechoice ===============================================================#
        output_json = []

        if dataset_category == 'multiple_choice':
            for question in tqdm(data):
                ex_output = {}
                labels,logits, raw_probs = [], [], []
                selected_dist=np.zeros(4)
                for ex in question:
                    print(prompt_template)

                    filled_prompt = prompt_template.format(
                        context=ex.context, 
                        question=ex.question
                    )
                    

                    print(filled_prompt)
                    import time; time.sleep(2)
                    for char, option in zip(string.ascii_uppercase, ex.options):
                        filled_prompt += f"{char}) {option}\n"

                    if 'llama' in system_name:
                        filled_prompt += '\nAnswer:'

                    print(filled_prompt)
                    
                    output = system.prompt_classifier_response(input_text=filled_prompt)
                    selected_dist[output.pred] += 1

                    logits += [output.logits]
                    raw_probs += [output.raw_probs]
                    labels.append(ex.label)

                ex_output['q_id'] = ex.ex_id.split('-')[0]
                ex_output['logits'] = np.array(logits)
                ex_output['raw_probs'] = np.array(raw_probs)
                ex_output['labels'] = labels
                output_json.append(ex_output)

            # quick eval
            acc = np.mean([np.argmax(ex['logits'][0]) == ex['labels'][0] for ex in output_json])
            print(f"{dataset:<15} p{p_num:<4} acc: {100*acc:.1f}")

        #== code for comparative ==================================================================#
        elif dataset_category == 'comparative':
            for c_num, doc in enumerate(tqdm(data)):
                ex_output = {}
                selected_dist = np.zeros(2)

                N = len(doc.responses)
                logits_matrix = np.zeros((N, N, 2))
                raw_probs_matrix = np.zeros((N, N, 2))
                for i in range(len(doc.responses)):
                    for j in range(len(doc.responses)):
                        if i==j: continue
                        response_1 = doc.responses[i]
                        response_2 = doc.responses[j]

                        filled_prompt = prompt_template.format(
                            context=doc.context, 
                            response_1=response_1,
                            response_2=response_2,
                            adjective=adjective
                        )
                        
                        output = system.prompt_classifier_response(input_text=filled_prompt)

                        logits_matrix[i, j] = output.logits
                        raw_probs_matrix[i, j] = output.raw_probs

                ex_output['c_num'] = doc.context_id
                ex_output['logits'] = logits_matrix
                ex_output['raw_probs'] = raw_probs_matrix
                ex_output['labels'] = doc.scores[score_type]
                output_json.append(ex_output)
        
        save_pickle(output_json, output_path)

def create_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--system-name', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--score-type', type=str, default=None, help='which transformer to use')
    parser.add_argument('--peft-path', type=str, default=None, help='weights to use (if provided)')
    parser.add_argument('--device', type=str, default=None, help='the device to use for generate')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    run_bias_search(
        dataset=args.dataset,
        system_name=args.system_name,
        score_type=args.score_type,
        peft_path=args.peft_path,
        device=args.device
    )

#python mcrc_bias.py --dataset race-M --system-name llama2-7b-chat --device cuda:3