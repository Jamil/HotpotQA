import json
import sys


with open("dev_distractor_pred.json") as f:
    x = json.loads(f.readline())


with open("result_dev_distractor_pred3.json") as f2:
    z = json.loads(f2.readline())

def check_if_date(answer):
    digits = []
    for a in answer:
        if a == '0' or a == '1' or a == '2' or a == '3' or a == '4' or a == '5' or a =='6' or a == '7' or a == '8' or a == '9':
            digits.append(a)
    if len(digits) >= 4:
        return True
    return False


# def check_if_similar_name(facts1, facts2):
#     if len(facts2) != len(facts1):
#         return False
#     for x,y in zip(sortedfacts1, facts2):




def compute_results():
    all_keys = []
    for key, value in x['sp'].items():
        all_keys.append(key)

    correct = []
    incorrect = []
    absent_keys = []
    correct_with_sp = []
    correct_without_sp = []
    index = 0
    date_issues = []
    number_of_results = sys.argv[2]
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
                index += 1
                incorrect.append(item_id)
                if check_if_date(item['answer']) is True:
                    date_issues.append(item['_id'])
                print ("===================================================================")
                print ("Question", item['question'], item['_id'])
                print ('Answer', item['answer'])
                print ("Gold SFacts", item['supporting_facts'])
                print ("Model answer", x['answer'][item_id])
                print ("Model SFacts", x['sp'][item_id])
                show_context = sys.argv[1]
                if show_context == 'True':
                    for c in item['context']:
                        print ('-------------------------------------------------------------------------------')
                        print ("Title", c[0])
                        print ("Content", c[1])
                if index == int(number_of_results):
                    break
        else:
            absent_keys.append(item_id)
    print ("date issues are", len(date_issues))


def show_all_results(question_id):
    for item in z:
        if item['_id'] == question_id:
            item_id = item['_id']
            print ("===================================================================")
            print ("Question", item['question'], item['_id'])
            print ('Answer', item['answer'])
            print ("Gold SFacts", item['supporting_facts'])
            print ("Model answer", x['answer'][item_id])
            print ("Model SFacts", x['sp'][item_id])
            for c in item['context']:
                print ('-------------------------------------------------------------------------------')
                print ("Title", c[0])
                print ("Content", c[1])

if __name__ == "__main__":
    compute_results()
    continue_or_not = raw_input("Do you want to continue")
    if continue_or_not == "Y" or continue_or_not == "y":
        question_id = raw_input("which id do you want to view")
        show_all_results(question_id)
