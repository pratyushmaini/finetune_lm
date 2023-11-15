from typing import Dict, List, Union

def multiple_choice(inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = '{query}\nOptions:{options}\nAnswer: '
    assert isinstance(inp['choices'], List)
    options = ''
    for option in inp['choices']:
        options += f'\n - {option}'
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query, options=options),
        'response': inp['choices'][inp['gold']],
    }

def question_answer(inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = '{query}\nAnswer: '
    assert isinstance(inp['choices'], List)
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query),
        'response': inp['choices'][inp['gold']],
    }

composer_dict = {
    "piqa": question_answer,
    "commonsense_qa": multiple_choice,
    "arc_easy": multiple_choice,
    "trivia_qa": multiple_choice,
    
}