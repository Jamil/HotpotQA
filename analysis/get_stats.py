import json
with open("sample_dev_pred.json") as f:
    x = json.loads(f.readline())

all_keys = []
for key, value in x['sp'].items():
    all_keys.append(key)

with open("result_dev_distractor_pred3.json") as f2:
    z = json.loads(f2.readline())
correct = []
incorrect = []
absent_keys = []
correct_with_sp = []
correct_without_sp = []
date_issues = []
incorrect_with_correct_sp = []
incorrect_with_incorrect_sp = []


for item in z:
    item_id = item['_id']

    if item_id in x['answer']:
        model_result = x['answer'][item_id]
        if model_result == item['answer']:
            correct.append(item_id)
            supporting_facts = item['supporting_facts']
            if supporting_facts == x['sp'][item_id]:
                correct_with_sp.append(item_id)
            else:
                correct_without_sp.append(item_id)
        else:
            # if check_if_data(item['answer']) is True:
            #     date_issues.append(item['_id'])
            incorrect.append(item_id)
            supporting_facts = item['supporting_facts']
            if supporting_facts == x['sp'][item_id]:
                incorrect_with_correct_sp.append(item_id)
            else:
                incorrect_with_incorrect_sp.append(item_id)

    else:
        absent_keys.append(item_id)
print ("Total correct (only answer)", len(correct))
print ("Total correct (answer + sp)", len(correct_with_sp))
print ("Total Correct (answer + sp wrong)", len(correct_without_sp))
print ("Total incorrect", len(incorrect))
print ("Incorrect answer but with correct sf", len(incorrect_with_correct_sp))
print ("incorrect answer and incorrect supporting fact", len(incorrect_with_incorrect_sp))
print ("Total documents graded", len(z))
# print ("show the sp", x['sp'])
