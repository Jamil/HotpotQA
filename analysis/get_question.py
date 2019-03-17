import sys
from textwrap import TextWrapper
from functools import reduce
from utils import *

DEFAULT_QUESTION_FILE = 'hotpot_dev_distractor_v1.json'

if __name__ == '__main__':
    question_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION_FILE
    id_question_map = read_questions(question_file)
    id = sys.argv[1]

    example = id_question_map[id]

    wrapper = TextWrapper(width=80,
                          initial_indent='\t',
                          subsequent_indent='\t')

    contexts = list(map(lambda x: x[1], example['context']))
    contexts = list(map(lambda x: reduce(lambda x, y: x + y, x), contexts))

    for i, context in enumerate(contexts):
        print(f'Context {i+1}')
        res = wrapper.fill(context)
        print(res)

    sup = list(map(lambda x: x[0], example['supporting_facts']))
    print(f'Question: {example["question"]}')
    print(f'Gold Ans: {example["answer"]}')
    print(f'Gold Sup: {sup}')

