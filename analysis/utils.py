import json
import string
import re
from collections import Counter

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

def check_sup_equal(sup1, sup2):
    sup1_entities_only = list(map(lambda x: x[0], sup1))
    sup2_entities_only = list(map(lambda x: x[0], sup2))
    sup1_set = set(sup1_entities_only)
    sup2_set = set(sup2_entities_only)
    return sup1_set == sup2_set

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
