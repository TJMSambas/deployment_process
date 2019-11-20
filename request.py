# %load request.py
import requests
import json

# URL
# url = 'http://127.0.0.1:5003/api'
url = 'http://rangga170397.pythonanywhere.com/api'

# Change the value of experience that you want to test
payload = {0 : {'account_check_status': '< 0 DM',
                 'duration_in_month': 18,
                 'credit_history': 'existing credits paid back duly till now',
                 'purpose': 'domestic appliances',
                 'credit_amount': 3190,
                 'savings': '... < 100 DM',
                 'present_emp_since': '1 <= ... < 4 years',
                 'installment_as_income_perc': 2,
                 'personal_status_sex': 'female : divorced/separated/married',
                 'other_debtors': 'none',
                 'present_res_since': 2,
                 'property': 'real estate',
                 'age': 24,
                 'other_installment_plans': 'none',
                 'housing': 'own',
                 'credits_this_bank': 1,
                 'job': 'skilled employee / official',
                 'people_under_maintenance': 1,
                 'telephone': 'none',
                 'foreign_worker': 'yes'}
          }


r = requests.post(url, json = payload)

print(r.json())