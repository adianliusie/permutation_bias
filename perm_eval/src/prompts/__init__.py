SCORE_TO_ADJECTIVE = {
    'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant', 
    'naturalness':'natural', 'continuity':'good a continuation', 'engagingness':'engaging', 
    'grammar':'grammatically correct', 'overall':'good', 'semantic':None, 'good': 'good'
}

def get_prompt_template(dataset:str):
    if dataset.endswith('-s'):
        dataset = dataset[:-2]
    
    if dataset in ['cosmos', 'race++', 'race', 'reclor']:
        prompt_template = '{context}\n{question}'
    elif dataset in ['arc-easy', 'arc-challenge', 'medmcqa']:
        prompt_template = '{question}'
    elif dataset in ['summeval']:
        prompt_template = '{context}\n\nWhich Summary is more {adjective}?'
    elif dataset in ['hellaswag']:
        prompt_template = '{context}'
    return prompt_template

def get_label_words(dataset:str):
    if dataset.endswith('-s'):
        dataset = dataset[:-2]
    
    if dataset in ['cosmos', 'race++', 'race', 'reclor', 'hellaswag', 'arc-easy', 'arc-challenge']:
        decoder_prefix = ''
        label_words = ['A', 'B', 'C', 'D']
    elif dataset in ['summeval']:
        decoder_prefix = 'Summary'
        label_words = ['A', 'B']
    return decoder_prefix, label_words
