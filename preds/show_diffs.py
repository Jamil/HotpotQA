import json
import sys

DEFAULT_QUESTION_FILE = 'hotpot_dev_distractor_v1.json'
DEFAULT_PREDICTION_DIR = 'preds'
DEFAULT_BASELINE_PREDICTION_FILE = 'baseline_pred.json'

def read_questions(fn):
    id_question_map = {}
    qs = json.loads(open(fn).read())
    for q in qs:
        id_question_map[q['_id']] = q
    return id_question_map

def read_predictions(fn):
    preds = json.loads(open(fn).read())
    ans = preds['answer']
    sp = preds['sp']
    return ans, sp

def find_mismatched_questions(ans1, sup1, ans2, sup2):
    # Get questions for which either the
    # answer or sup predictions mismatch
    mismatched_qs =  [k for k in ans1.keys() if ans1[k] != ans2[k]]
    mismatched_qs += [k for k in sup1.keys() if sup1[k] != sup2[k]]
    return mismatched_qs

def print_mismatched_questions(f1, f2, id_question_map, limit=-1):
    ans1, sup1 = read_predictions(f1)
    ans2, sup2 = read_predictions(f2)
    mismatched_qs = find_mismatched_questions(ans1, sup1, ans2, sup2)

    limit = limit if limit != -1 else len(mismatched_qs)
    for q_id in mismatched_qs:
        question = id_question_map[q_id]
        a1, a2 = ans1[q_id], ans2[q_id]
        s1, s2 = sup1[q_id], sup2[q_id]
        print(f'{question["question"]}')
        print(f'Answer  : {question["answer"]}')
        print(f'Answer A: {a1}')
        print(f'Answer B: {a2}')
        print(f'Sup     : {str(question["supporting_facts"])}')
        print(f'Sup    A: {str(s1)}')
        print(f'Sup    B: {str(s2)}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Run python show_diffs.py preds1.json preds2.json ' + \
              '{hotpot_dev_distractor_v1.json}')
        sys.exit()

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    question_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_QUESTION_FILE
    
    id_question_map = read_questions(question_file)
    print_mismatched_questions(file1, file2, id_question_map, limit=100)
