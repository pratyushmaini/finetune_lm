from typing import Dict, List, Union



def multiple_choice(inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = 'Question: {query}\nOptions:{options}\nAnswer:'
    assert isinstance(inp['choices'], List)
    options = ''
    for i, option in enumerate(inp['choices']):
        char = chr(ord('A') + i)
        # options += f' {char}. {option}\n'
        options += f'\n - {option}'
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query, options=options),
        'response': f" {inp['choices'][inp['gold']]}",
    }

def question_answer(inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = 'Question: {query}\nAnswer:'
    assert isinstance(inp['choices'], List)
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query),
        'response': f" {inp['choices'][inp['gold']]}",
    }

composer_dict = {
    "piqa": question_answer,
    "commonsense_qa": question_answer,
    "arc_easy": question_answer,
    "trivia_qa": question_answer,

}