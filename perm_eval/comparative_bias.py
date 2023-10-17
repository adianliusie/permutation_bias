import os
import random
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict

from src.data import load_nlg_data
from src.models.llama import PromptedLlama2System
from src.models.flant5 import PromptedFlanT5System
from src.utils.general import save_pickle

from src.prompts import SCORE_TO_ADJECTIVE, COMPARATIVE_TEMPLATES
# PROMPT_TEMPLATES = [
#     '{context}\n\nWhich Summary is more {adjective}?\n\nSummary A: {response_1}\nSummary B: {response_2}',
#     '{context}\n\nSummary A: {response_1}\nSummary B: {response_2}\n\nWhich Summary is more {adjective} with respect to the article, Summary A or Summary B?',
#     'Assess the following two summaries given the corresponding passage, and determine which summary is more {adjective}.\n\nPassage:\n{context}\n\nSummary A: {response_1}\nSummary B: {response_2}\n\nWhich Summary is more {adjective}?',
# ]

# SCORE_TO_ADJECTIVE = {
#     'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant', 
#     'naturalness':'natural', 'continuity':'good a continuation', 'engagingness':'engaging', 
#     'grammar':'grammatically correct', 'overall':'good', 'semantic':None, 'good': 'good'
# }

def run_comparative_bias_search(
    dataset:str,
    score_type:str,
    system_name:str,
    device:str='cuda'
):
    #== define prompt templates for task ======================================================================#
    adjective = SCORE_TO_ADJECTIVE[score_type]
    prompt_templates = COMPARATIVE_TEMPLATES
        
    if ('llama' in system_name) or ('vicuna' in system_name):
        prompt_templates = [template + '\n\nAnswer:' for template in prompt_templates]
    
    label_words = ['A', 'B']
    decoder_prefix='Summary'    
    
    #== prepare comaprative assessment dataset ========================================================#
    data = load_nlg_data(dataset)

    #== load LLM system ==============================================================================#
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
    
    #== Make Dir if not already exists ===============================================================#
    if not os.path.isdir(f"outputs/{dataset}"):
        os.makedirs(f"outputs/{dataset}")
    
    #== Go through all selected examples =============================================================#
    for p_num, prompt_template in enumerate(prompt_templates):
        output_path = f"outputs/{dataset}/{score_type}_{system_name}_p{p_num}.pk"
        if os.path.isfile(output_path):
            continue

        output_json = []
        for c_num, doc in enumerate(tqdm(data)):
            ex_output = {}
            detailed, hits, selected_dist= [], [], np.zeros(2)

            N = len(doc.responses)
            logits_matrix = np.zeros((N, N, 2))
            raw_probs_matrix = np.zeros((N, N, 2))
            selected_dist = np.zeros(2)
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

                    selected_dist[output.pred] += 1

            ex_output['c_num'] = doc.context_id
            ex_output['logits'] = logits_matrix
            ex_output['raw_probs'] = raw_probs_matrix
            ex_output['labels'] = doc.scores[score_type]
            output_json.append(ex_output)

        save_pickle(output_json, output_path)

def create_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='summeval', help='dataset to use')
    parser.add_argument('--system-name', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--score-type', type=str, default='relevance', help='score type to use')
    parser.add_argument('--device', type=str, default=None, help='the device to use for generate')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    run_comparative_bias_search(
        dataset=args.dataset,
        system_name=args.system_name,
        score_type=args.score_type,
        device=args.device
    )

#python mcrc_bias.py --dataset race-M --system-name llama2-7b-chat --device cuda:3