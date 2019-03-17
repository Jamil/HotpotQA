import random
import sys

from utils import *

DEFAULT_QUESTION_FILE = 'hotpot_dev_distractor_v1.json'
DEFAULT_PREDICTION_DIR = 'preds'
DEFAULT_BASELINE_PREDICTION_FILE = 'baseline_pred.json'

def fetch_examples(ans1, sup1, ans2, sup2, id_question_map):
    baseline_f1_greater = []
    model_f1_greater = []
    baseline_model_same = []

    # go through all questions
    for q_id in ans1.keys():
        question_dict = id_question_map[q_id]
        question = question_dict['question']
        answer = question_dict['answer']
        a1, a2 = ans1[q_id], ans2[q_id]
        a1_f1 = f1_score(a1, answer)
        a2_f1 = f1_score(a2, answer)
        tup = (question, answer, a1, a2, q_id)
        if a1_f1 > a2_f1:
            baseline_f1_greater.append(tup)
        elif a2_f1 > a1_f1:
            model_f1_greater.append(tup)
        elif a1_f1 == a2_f1:
            baseline_model_same.append(tup)

    return (baseline_f1_greater, model_f1_greater, baseline_model_same)

def print_summary_and_examples(f1, f2, id_question_map, limit=3):
    ans1, sup1 = read_predictions(f1)
    ans2, sup2 = read_predictions(f2)

    baseline_f1_greater, model_f1_greater, baseline_model_same = fetch_examples(ans1, sup1, ans2, sup2, id_question_map)

    print(f'Baseline F1 > Model F1: {len(baseline_f1_greater)}')
    print(f'Model F1 < Baseline F1: {len(model_f1_greater)}')
    print(f'Baseline F1 = Model F1: {len(baseline_model_same)}')

    random.shuffle(baseline_f1_greater)
    random.shuffle(model_f1_greater)
    random.shuffle(baseline_model_same)

    print('------------------------------------')
    print('Examples from Baseline F1 > Model F1')
    print('------------------------------------')

    for ex in baseline_f1_greater[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')

    print('------------------------------------')
    print('Examples from Model F1 > Baseline F1')
    print('------------------------------------')

    for ex in model_f1_greater[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')

    print('------------------------------------')
    print('Examples from Baseline F1 = Model F1')
    print('------------------------------------')

    for ex in baseline_model_same[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run python ans_comparison.py preds1.json preds2.json ' + \
              '{hotpot_dev_distractor_v1.json}')
        sys.exit()

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    question_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_QUESTION_FILE
    
    id_question_map = read_questions(question_file)
    print_summary_and_examples(file1, file2, id_question_map, limit=3)
