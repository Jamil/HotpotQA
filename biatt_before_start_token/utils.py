import json

def read_questions(fn):
    id_question_map = {}
    qs = json.loads(open(fn).read())
    for q in qs:
        id_question_map[q['_id']] = q
    return id_question_map

def read_predictions(fn):
    preds = json.loads(open(fn).read())
    print ("show +++++", type(preds))
    ans = preds['answer']
    sp = preds['sp']
    return ans, sp

def check_sup_equal(sup1, sup2):
    sup1_entities_only = list(map(lambda x: x[0], sup1))
    sup2_entities_only = list(map(lambda x: x[0], sup2))
    sup1_set = set(sup1_entities_only)
    sup2_set = set(sup2_entities_only)
    return sup1_set == sup2_set
