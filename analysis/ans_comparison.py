import random
import sys

from utils import *

DEFAULT_QUESTION_FILE = 'hotpot_dev_distractor_v1.json'
DEFAULT_PREDICTION_DIR = 'preds'
DEFAULT_BASELINE_PREDICTION_FILE = 'baseline_pred.json'

def fetch_examples(ans1, sup1, ans2, sup2, id_question_map):
    baseline_correct_model_wrong = []
    baseline_correct_model_correct = []
    baseline_wrong_model_correct = []
    baseline_wrong_model_wrong = []

    # go through all questions
    for q_id in ans1.keys():
        question_dict = id_question_map[q_id]
        question = question_dict['question']
        answer = question_dict['answer']
        a1, a2 = ans1[q_id], ans2[q_id]
        if a1 == answer and a2 == answer:
            baseline_correct_model_correct.append((question, answer, a1, a2, q_id))
        elif a1 == answer and a2 != answer:
            baseline_correct_model_wrong.append((question, answer, a1, a2, q_id))
        elif a1 != answer and a2 == answer:
            baseline_wrong_model_correct.append((question, answer, a1, a2, q_id))
        elif a1 != answer and a2 != answer:
            baseline_wrong_model_wrong.append((question, answer, a1, a2, q_id))
    return (baseline_correct_model_wrong, baseline_correct_model_correct, baseline_wrong_model_correct, baseline_wrong_model_wrong)

def print_summary_and_examples(f1, f2, id_question_map, limit=3):
    ans1, sup1 = read_predictions(f1)
    ans2, sup2 = read_predictions(f2)

    baseline_correct_model_wrong, baseline_correct_model_correct, baseline_wrong_model_correct, baseline_wrong_model_wrong = fetch_examples(ans1, sup1, ans2, sup2, id_question_map)

    print(f'Baseline CORRECT, Model CORRECT: {len(baseline_correct_model_correct)}')
    print(f'Baseline WRONG,   Model CORRECT: {len(baseline_wrong_model_correct)}')
    print(f'Baseline CORRECT, Model WRONG  : {len(baseline_correct_model_wrong)}')
    print(f'Baseline WRONG,   Model WRONG  : {len(baseline_wrong_model_wrong)}')

    random.shuffle(baseline_correct_model_wrong)
    random.shuffle(baseline_wrong_model_wrong)
    random.shuffle(baseline_correct_model_correct)
    random.shuffle(baseline_wrong_model_correct)

    print('------------------------------------------')
    print('Examples from Baseline CORRECT, Model CORRECT')
    print('------------------------------------------')

    for ex in baseline_correct_model_correct[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')

    print('------------------------------------------')
    print('Examples from Baseline WRONG, Model CORRECT')
    print('------------------------------------------')

    for ex in baseline_wrong_model_correct[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')

    print('------------------------------------------')
    print('Examples from Baseline CORRECT, Model WRONG')
    print('------------------------------------------')

    for ex in baseline_correct_model_wrong[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')

    print('------------------------------------------')
    print('Examples from Baseline WRONG, Model WRONG')
    print('------------------------------------------')

    for ex in baseline_wrong_model_wrong[:limit]:
        print(f'\tQuestion:     {ex[0]}')
        print(f'\t              ({ex[4]})')
        print(f'\tGold Ans:     {ex[1]}')
        print(f'\tBaseline Ans: {ex[2]}')
        print(f'\tModel A Ans:  {ex[3]}')
        print(f'\t---')

    # write out answers
    dict_to_write = {
        'baseline_correct_model_wrong': baseline_correct_model_wrong,
        'baseline_correct_model_correct': baseline_correct_model_correct,
        'baseline_wrong_model_correct': baseline_wrong_model_correct,
        'baseline_wrong_model_wrong': baseline_wrong_model_wrong
    }
    json_to_write = json.dumps(dict_to_write)
    f = open('preds/ans_analysis.json', 'w')
    f.write(json_to_write)

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
