from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

BASE_PATH = './serialization/'

clf_budget_category = pickle.load(open(BASE_PATH + '/pipeline_budget_category.pkl', 'rb'))

clf_income_tags = pickle.load(open(BASE_PATH + '/pipeline_tags_income.pkl', 'rb'))
clf_wants_tags = pickle.load(open(BASE_PATH + '/pipeline_tags_wants.pkl', 'rb'))
clf_savings_tags = pickle.load(open(BASE_PATH + '/pipeline_tags_savings.pkl', 'rb'))
clf_needs_tags = pickle.load(open(BASE_PATH + '/pipeline_tags_needs.pkl', 'rb'))

LABELS_BUDGET_CATEGORY = {0: 'needs', 1: 'wants', 2: 'savings'}

LABELS_INCOME = {0: 'interests', 1: 'capitalGains', 2: 'salary', 3: 'dividends', 4: 'pension', 5: 'realEstate'}
LABELS_SAVINGS = {0: 'certificateOfDeposit', 1: 'stocks', 2: 'pension', 3: 'cryptosAndNFTs'}
LABELS_WANTS = {0: 'food', 1: 'entertainment', 2: 'clothing', 3: 'shopping', 4: 'subscriptions', 5: 'gym', 6: 'parties', 7: 'trips'}
LABELS_NEEDS = {0: 'taxes', 1: 'internet', 2: 'transportation', 3: 'phone', 4: 'utilities', 5: 'education', 6: 'groceries', 7: 'housing', 8: 'insurance', 9: 'healthCare', 10: 'pets'}

def get_predicted_tag(description, clf, labels):
    index = clf.predict([description])[0]

    predicted_tag = labels[index]
    proba = clf.predict_proba([description])[0][index]

    if proba < 0.5: predicted_tag = 'others'

    return predicted_tag

def complete_transaction(transaction, predicted_bc, predicted_tag):
    transaction['description'] = transaction['description'].upper()
    transaction['budget_category'] = predicted_bc
    transaction['tag'] = predicted_tag

    return transaction

def process_transactions(transactions):
    for i in range(len(transactions)):
        transaction = transactions[i]

        transaction_type = transaction['type'].upper()
        transaction_description = transaction['description'].upper()

        if transaction_type == 'INFLOW':
            predicted_tag = get_predicted_tag(transaction_description, clf_income_tags, LABELS_INCOME)

            transaction = complete_transaction(transaction, 'income', predicted_tag)
        else:
            predicted_bc = clf_budget_category.predict([transaction_description])
            predicted_bc = LABELS_BUDGET_CATEGORY[predicted_bc[0]]

            if predicted_bc == 'savings':
                predicted_tag = get_predicted_tag(transaction_description, clf_savings_tags, LABELS_SAVINGS)
            elif predicted_bc == 'needs':
                predicted_tag = get_predicted_tag(transaction_description, clf_needs_tags, LABELS_NEEDS)
            elif predicted_bc == 'wants':
                predicted_tag = get_predicted_tag(transaction_description, clf_wants_tags, LABELS_WANTS)
            else:
                predicted_tag = 'error'

            transaction = complete_transaction(transaction, predicted_bc, predicted_tag)

        transactions[i] = transaction

    return transactions

@app.route('/categorizer', methods=['POST'])
def categorizer():
    transactions = list(request.json)    
    return jsonify(process_transactions(transactions))

if __name__ == "__main__":
    app.run()
