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
model = pickle.load(open('LogReg.pkl','rb'))

# Load the scaler & encoder
scaler = pickle.load(open('scaler.pkl','rb'))
dict_encoder = pickle.load(open('label_dictionary.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force = True)
    
    # ---------------DEVELOPER ZONE---------------------
    # Defining parameter
    include_categorical = True
    num_predictors = ['credit_amount',
                     'installment_as_income_perc',
                     'credits_this_bank',
                     'present_res_since',
                     'age',
                     'people_under_maintenance',
                     'duration_in_month']
    if include_categorical:
        cat_predictors = ['job',
                         'present_emp_since',
                         'other_installment_plans',
                         'credit_history',
                         'personal_status_sex',
                         'foreign_worker',
                         'other_debtors',
                         'telephone',
                         'savings',
                         'property',
                         'purpose',
                         'account_check_status',
                         'housing']
    
    # Data preprocessing
    df = pd.DataFrame.from_dict(data, orient = 'index')
    df_numerical = scaler.transform(df[num_predictors])
    if include_categorical:
        df_categorical = df[cat_predictors].apply(lambda x: dict_encoder[x.name].transform(x)).values
        data_input = np.concatenate((df_numerical, df_categorical), axis = 1)
    else:
        data_input = df_numerical
    # ---------------(END) DEVELOPER ZONE---------------------
    
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data_input)

    # Take the first value of prediction
    output = prediction.tolist()
    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=None)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")