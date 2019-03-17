import json
import sys
import random

from utils import *

DEFAULT_QUESTION_FILE = 'hotpot_dev_distractor_v1.json'
DEFAULT_PREDICTION_DIR = 'preds'
DEFAULT_BASELINE_PREDICTION_FILE = 'baseline_pred.json'

def find_mismatched_questions(ans1, sup1, ans2, sup2):
    # Get questions for which either the
    # answer or sup predictions mismatch
    mismatched_qs =  [k for k in ans1.keys() if ans1[k] != ans2[k]]
    mismatched_qs += [k for k in sup1.keys() if not check_sup_equal(sup1[k], sup2[k])]
    return mismatched_qs

def print_mismatched_questions(f1, f2, id_question_map, limit=-1, randomize=False):
    ans1, sup1 = read_predictions(f1)
    ans2, sup2 = read_predictions(f2)
    mismatched_qs = find_mismatched_questions(ans1, sup1, ans2, sup2)

    qs_to_enumerate = mismatched_qs
    if randomize:
        qs_to_enumerate = random.shuffle(mismatched_qs)
    if limit != -1:
        qs_to_enumerate = mismatched_qs[:limit]

    for q_id in qs_to_enumerate:
        question = id_question_map[q_id]
        a1, a2 = ans1[q_id], ans2[q_id]
        s1, s2 = sup1[q_id], sup2[q_id]
        print('-' * 10)
        print(f'({str(q_id)}) {question["question"]}')
        print(f'Answer  : {question["answer"]}')
        print(f'Answer A: {a1}')
        print(f'Answer B: {a2}')
        print(f'Sup     : {str(question["supporting_facts"])}')
        print(f'Sup    A: {str(s1)}')
        print(f'Sup    B: {str(s2)}')
        print('-' * 10)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run python show_diffs.py preds1.json preds2.json ' + \
              '{hotpot_dev_distractor_v1.json}')
        sys.exit()

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    question_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_QUESTION_FILE
    
    id_question_map = read_questions(question_file)
    print_mismatched_questions(file1, file2, id_question_map, limit=10, randomize=True)
