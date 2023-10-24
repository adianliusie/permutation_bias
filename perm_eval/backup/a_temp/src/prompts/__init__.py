MCRC_TEMPLATES = [
    '{context}\n\n{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}',
    'Context: {context}\n\nQuestion:{question}\nA:{option_1}\nB:{option_2}\nC:{option_3}\nD:{option_4}',
    'Content{context}\n\nQuestion:{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}\n\nWhich option best answers the question: A, B, C or D?'
]


SCORE_TO_ADJECTIVE = {
    'consistency':'consistent', 'coherency':'coherent', 'fluency':'fluent', 'relevance':'relevant', 
    'naturalness':'natural', 'continuity':'good a continuation', 'engagingness':'engaging', 
    'grammar':'grammatically correct', 'overall':'good', 'semantic':None, 'good': 'good'
}

COMPARATIVE_TEMPLATES = [
    '{context}\n\nWhich Summary is more {adjective}?\n\nSummary A: {response_1}\nSummary B: {response_2}',
    '{context}\n\nSummary A: {response_1}\nSummary B: {response_2}\n\nWhich Summary is more {adjective} with respect to the article, Summary A or Summary B?',
    'Assess the following two summaries given the corresponding passage, and determine which summary is more {adjective}.\n\nPassage:\n{context}\n\nSummary A: {response_1}\nSummary B: {response_2}\n\nWhich Summary is more {adjective}?',
]

STORY_COMPLETION_TEMPLATES = [
        'Complete the following with the best ending choice:\n\n{context}\n\n\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}',
        'Select which of the following options best completes the context\n\nContext: {context}\n\nOptions:\nA:{option_1}\nB:{option_2}\nC:{option_3}\nD:{option_4}',
        '{context}\n\nWhich option best completes the previous context?\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}'
 ]


