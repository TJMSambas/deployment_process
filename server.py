# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model = pickle.load(open('lgbm.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Base input.
    base = pd.DataFrame.from_dict({ 0 : {'duration_in_month': 0,
                                     'credit_amount': 0,
                                     'installment_as_income_perc': 0,
                                     'present_res_since': 0,
                                     'age': 0,
                                     'credits_this_bank': 0,
                                     'people_under_maintenance': 0,
                                     'account_check_status_high': 0,
                                     'account_check_status_low': 0,
                                     'account_check_status_negative': 0,
                                     'account_check_status_no': 0,
                                     'credit_history_A': 0,
                                     'credit_history_B': 0,
                                     'credit_history_C': 0,
                                     'credit_history_D': 0,
                                     'credit_history_E': 0,
                                     'purpose_business': 0,
                                     'purpose_domestic': 0,
                                     'purpose_education': 0,
                                     'purpose_furniture': 0,
                                     'purpose_new_car': 0,
                                     'purpose_radtel': 0,
                                     'purpose_repairs': 0,
                                     'purpose_retraining': 0,
                                     'purpose_used_car': 0,
                                     'purpose_vacation': 0,
                                     'savings_high': 0,
                                     'savings_low': 0,
                                     'savings_medium': 0,
                                     'savings_no': 0,
                                     'savings_very_high': 0,
                                     'present_emp_since_long': 0,
                                     'present_emp_since_short': 0,
                                     'present_emp_since_unemployed': 0,
                                     'present_emp_since_very short': 0,
                                     'present_emp_since_very_long': 0,
                                     'personal_status_sex_ds_male': 0,
                                     'personal_status_sex_female': 0,
                                     'personal_status_sex_mw_male': 0,
                                     'personal_status_sex_single_male': 0,
                                     'other_debtors_co_applicant': 0,
                                     'other_debtors_guarantor': 0,
                                     'other_debtors_none': 0,
                                     'property_building': 0,
                                     'property_car': 0,
                                     'property_real_estate': 0,
                                     'property_unknown': 0,
                                     'other_installment_plans_bank': 0,
                                     'other_installment_plans_none': 0,
                                     'other_installment_plans_stores': 0,
                                     'housing_free': 0,
                                     'housing_own': 0,
                                     'housing_rent': 0,
                                     'telephone_none': 0,
                                     'telephone_registered': 0,
                                     'foreign_worker_no': 0,
                                     'foreign_worker_yes': 0}}, orient = 'index')
    
    # Get the data from the POST request.
    data = request.get_json(force = True)
    df = pd.DataFrame.from_dict(data, orient = 'index')

    # Feature engineering for categorical variables.
    df['account_check_status'] = df['account_check_status'].str.strip().map({'no checking account' : 'no',
                                                                            '< 0 DM' : 'negative',
                                                                            '0 <= ... < 200 DM' : 'low',
                                                                            '>= 200 DM / salary assignments for at least 1 year' : 'high'})

    df['credit_history'] = df['credit_history'].str.strip().map({'existing credits paid back duly till now' : 'A',
                                                                'critical account/ other credits existing (not at this bank)' : 'B',
                                                                'delay in paying off in the past' : 'C',
                                                                'all credits at this bank paid back duly' : 'D',
                                                                'no credits taken/ all credits paid back duly' : 'E'}) 

    df['purpose'] = df['purpose'].str.strip().map({'domestic appliances' : 'domestic',
                                                  'car (new)' : 'new_car',
                                                  'radio/television' : 'radtel',
                                                  'car (used)' : 'used_car',
                                                  '(vacation - does not exist?)' : 'vacation',
                                                  'furniture/equipment' : 'furniture',
                                                  'business' : 'business',
                                                  'education' : 'education',
                                                  'repairs' : 'repairs',
                                                  'retraining' : 'retraining'})

    df['savings'] = df['savings'].str.strip().map({'... < 100 DM' : 'low',
                                                  'unknown/ no savings account' : 'no',
                                                  '100 <= ... < 500 DM' : 'medium',
                                                  '500 <= ... < 1000 DM' : 'high', 
                                                  '.. >= 1000 DM' : 'very_high'})

    df['present_emp_since'] = df['present_emp_since'].str.strip().map({'... < 1 year' : 'very short',
                                                                      '1 <= ... < 4 years' : 'short',
                                                                      '4 <= ... < 7 years' : 'long',
                                                                      '.. >= 7 years' : 'very_long',
                                                                      'unemployed' : 'unemployed'})

    df['personal_status_sex'] = df['personal_status_sex'].str.strip().map({'male : single' : 'single_male',
                                                                          'female : divorced/separated/married' : 'female',
                                                                          'male : married/widowed' : 'mw_male',
                                                                          'male : divorced/separated' : 'ds_male'})

    df['other_debtors'] = df['other_debtors'].str.strip().map({'co-applicant' : 'co_applicant',
                                                              'none' : 'none',
                                                              'guarantor' : 'guarantor'})

    df['property'] = df['property'].str.strip().map({'if not A121/A122 : car or other, not in attribute 6' : 'car',
                                                    'real estate' : 'real_estate',
                                                    'if not A121 : building society savings agreement/ life insurance' : 'building',
                                                    'unknown / no property' : 'unknown'})

    df['other_installment_plans'] = df['other_installment_plans'].str.strip()

    df['housing'] = df['housing'].str.strip().map({'for free' : 'free',
                                                  'own' : 'own',
                                                  'rent' : 'rent'})

    df['job'] = df['housing'].str.strip().map({'skilled employee / official' : 'skilled',
                                              'unskilled - resident' : 'unskilled_res',
                                              'management/ self-employed/ highly qualified employee/ officer' : 'manager',
                                              'unemployed/ unskilled - non-resident' : 'unskilled_nores'})

    df['telephone'] = df['telephone'].str.strip().map({'yes, registered under the customers name' : 'registered',
                                                      'none' : 'none'})


    del df['job']

    df = pd.get_dummies(df)

    # Creating input
    columns = list(set(base.columns) - set(df.columns))
    data_input = pd.concat([df, base[columns]], axis = 1)
    data_input = data_input[list(base.columns)]    

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data_input)

    # Take the first value of prediction
    output = prediction[0].to_list()

    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run(port=5003, debug=None)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")


# In[ ]:




