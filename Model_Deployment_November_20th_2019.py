#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[229]:


df = pd.read_csv('D:/Titip/german_credit.csv')


# In[230]:


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


# In[231]:


from sklearn.model_selection import train_test_split

X = df.copy()
del X['default']
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# In[232]:


for i in X_train.columns :
    X_train[i] = X_train[i].astype(int)


# In[233]:


for i in X_test.columns :
    X_test[i] = X_test[i].astype(int)


# In[234]:


import lightgbm

train_data = lightgbm.Dataset(X_train, label = y_train)
test_data = lightgbm.Dataset(X_test, label = y_test)
parameters_lgb = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
model = lightgbm.train(params=parameters_lgb,
                       train_set= train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# In[192]:


df2 = df.copy()


# In[193]:


df2 = pd.get_dummies(df2)


# In[194]:


from bayes_opt import BayesianOptimization
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
import lightgbm as lgb


# In[195]:


bayesian_tr_idx = X_train.index
bayesian_val_idx = X_test.index


# In[197]:


bounds_LGB = {
    'num_leaves': (31, 500), 
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.01, 0.3),
    'min_child_weight': (0.00001, 0.01),   
    'reg_alpha': (1, 2), 
    'reg_lambda': (1, 2),
    'max_depth':(-1,50),
}


# In[198]:


def LGB_bayesian(
    learning_rate,
    num_leaves, 
    bagging_fraction,
    feature_fraction,
    min_child_weight, 
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    

    param = {
              'num_leaves': num_leaves, 
              'min_data_in_leaf': min_data_in_leaf,
              'min_child_weight': min_child_weight,
              'bagging_fraction' : bagging_fraction,
              'feature_fraction' : feature_fraction,
              'learning_rate' : learning_rate,
              'max_depth': max_depth,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'objective': 'binary',
              'save_binary': True,
              'seed': 1337,
              'feature_fraction_seed': 1337,
              'bagging_seed': 1337,
              'drop_seed': 1337,
              'data_random_seed': 1337,
              'boosting_type': 'gbdt',
              'verbose': 1,
              'is_unbalance': False,
              'boost_from_average': True,
              'metric':'auc'}    
    
    oof = np.zeros(len(df2))
    trn_data= lgb.Dataset(df2.iloc[bayesian_tr_idx][features].values, label=df2.iloc[bayesian_tr_idx][target].values)
    val_data= lgb.Dataset(df2.iloc[bayesian_val_idx][features].values, label=df2.iloc[bayesian_val_idx][target].values)

    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)
    
    oof[bayesian_val_idx]  = clf.predict(df2.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  
    
    score = roc_auc_score(df2.iloc[bayesian_val_idx][target].values, oof[bayesian_val_idx])

    return score


# In[199]:


LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=0)


# In[200]:


init_points = 10
n_iter = 15


# In[202]:


import warnings
warnings.filterwarnings("ignore")
features = list(df2)
features.remove('default')
target = 'default'

print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[203]:


param_lgb = {
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
        'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_child_weight': LGB_BO.max['params']['min_child_weight'],
        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'reg_lambda': LGB_BO.max['params']['reg_lambda'],
        'reg_alpha': LGB_BO.max['params']['reg_alpha'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), 
        'objective': 'binary',
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'boosting_type': 'gbdt',
        'verbose': 1,
        'is_unbalance': False,
        'boost_from_average': True,
        'metric':'auc'
    }


# In[205]:


import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[208]:


from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from scipy import interp
import itertools

plt.rcParams["axes.grid"] = True

nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)

oof = np.zeros(len(df2))
mean_fpr = np.linspace(0,1,100)
cms= []
tprs = []
aucs = []
y_real = []
y_proba = []
recalls = []
roc_aucs = []
f1_scores = []
accuracies = []
precisions = []
predictions = np.zeros(len(X_test))
feature_importance_df2 = pd.DataFrame()

i = 1
for train_idx, valid_idx in skf.split(df2, df2['default'].values):
    print("\nfold {}".format(i))
    trn_data = lgb.Dataset(df2.iloc[train_idx][features].values,
                                   label=df2.iloc[train_idx][target].values
                                   )
    val_data = lgb.Dataset(df2.iloc[valid_idx][features].values,
                                   label=df2.iloc[valid_idx][target].values
                                   )   
    
    clf = lgb.train(param_lgb, trn_data, num_boost_round = 500, valid_sets = [trn_data, val_data], verbose_eval = 100, early_stopping_rounds = 100)
    oof[valid_idx] = clf.predict(df2.iloc[valid_idx][features].values) 
    
    predictions += clf.predict(X_test[features]) / nfold
    
    # Scores 
    roc_aucs.append(roc_auc_score(df2.iloc[valid_idx][target].values, oof[valid_idx]))
    accuracies.append(accuracy_score(df2.iloc[valid_idx][target].values, oof[valid_idx].round()))
    recalls.append(recall_score(df2.iloc[valid_idx][target].values, oof[valid_idx].round()))
    precisions.append(precision_score(df2.iloc[valid_idx][target].values ,oof[valid_idx].round()))
    f1_scores.append(f1_score(df2.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Roc curve by folds
    f = plt.figure(1)
    fpr, tpr, t = roc_curve(df2.iloc[valid_idx][target].values, oof[valid_idx])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
    
    # Precion recall by folds
    g = plt.figure(2)
    precision, recall, _ = precision_recall_curve(df2.iloc[valid_idx][target].values, oof[valid_idx])
    y_real.append(df2.iloc[valid_idx][target].values)
    y_proba.append(oof[valid_idx])
    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  
    
    i= i+1
    
    # Confusion matrix by folds
    cms.append(confusion_matrix(df2.iloc[valid_idx][target].values, oof[valid_idx].round()))
    
    # Features imp
    fold_importance_df2 = pd.DataFrame()
    fold_importance_df2["Feature"] = features
    fold_importance_df2["importance"] = clf.feature_importance()
    fold_importance_df2["fold"] = nfold + 1
    feature_importance_df2 = pd.concat([feature_importance_df2, fold_importance_df2], axis=0)

# Metrics
print(
        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),
        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),
        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),
        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),
        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))
)

#ROC 
f = plt.figure(1)
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LGB ROC curve by folds')
plt.legend(loc="lower right")

# PR plt
g = plt.figure(2)
plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
plt.plot(recall, precision, color='blue',
         label=r'Mean P|R')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P|R curve by folds')
plt.legend(loc="lower left")

# Confusion maxtrix & metrics
plt.rcParams["axes.grid"] = False
cm = np.average(cms, axis=0)
default_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=default_names, 
                      title= 'LGB Confusion matrix [averaged/folds]')
plt.show()


# In[209]:


LGB_BO.max['params']

