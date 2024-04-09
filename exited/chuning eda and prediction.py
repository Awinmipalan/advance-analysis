#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


rawdata=pd.read_csv('C:\\Users\\DELL\\Downloads\\exited\\train.csv')


# In[3]:


rawdata


# In[4]:


df=rawdata.copy()


# In[5]:


df=df.drop(["id","CustomerId", "Surname"], axis = 1)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.describe().T


# In[9]:


df.isna().sum()


# In[10]:


num_col = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
cat_col = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']


# In[11]:


for col in num_col:
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    sns.histplot(df[col], ax=ax[0])  
    sns.boxplot(x=df[col], ax=ax[1])


# In[12]:


for feature in cat_col:
    if feature == "Exited":
        continue
    
    plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=feature, hue='Exited')
    plt.show()


# In[13]:


from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
df['Geography'] = le1.fit_transform(df['Geography'])

le2 = LabelEncoder()
df['Gender'] = le2.fit_transform(df['Gender'])

df.head()


# In[14]:


plt.figure(figsize=(10,10))
sns.heatmap(df.drop("Exited", axis=1).corr(), annot = True)


# In[15]:


plt.figure(figsize=(10,10))
sns.heatmap(df.drop("Exited", axis=1).cov(), annot = True)


# In[16]:


X = df.drop("Exited", axis=1)
y = df.Exited


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# models
models = {
    'LogisticRegression': LogisticRegression(),
    'NaiveBayes': GaussianNB(),
    'LDA': LDA(),
    'kNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
}

model_scores = {}

# Perform grid search for selected models
for model_name, model in models.items():    
    model.fit(X_train, y_train)
  
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    auc_score = roc_auc_score(y_test, y_pred_proba)
    model_scores[model_name] = auc_score

    print(f"Test AUC for {model_name}:", auc_score)
    print()
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette="viridis")
plt.title('Model AUC ROC Scores')
plt.ylabel('AUC ROC Score')
plt.show()


# In[19]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# # Parameter grid for RandomizedSearchCV
# random_param_grid = {
#     'n_estimators': np.arange(100, 1001, 100),
#     'max_features': ['auto', 'sqrt'],
#     'max_depth': np.arange(10, 101, 10),
#     'min_samples_split': np.arange(2, 11, 2),
#     'min_samples_leaf': np.arange(1, 5),
#     'bootstrap': [True, False]
# }

# # Initialize the classifier
# clf = RandomForestClassifier()

# # Randomized search on hyper parameters
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=random_param_grid,
#                                    n_iter=60, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='roc_auc')

# # Fit the random search model
# random_search.fit(X_train, y_train)

# # Best parameters from random search
# best_params_random = random_search.best_params_
# print("Best parameters found by RandomizedSearchCV:", best_params_random)

best_paras = {'n_estimators': 1000, 'min_samples_split': 8, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}

best_model_rf = RandomForestClassifier(**best_paras)
best_model_rf.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = best_model_rf.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC of the best model RF on the test set:", auc)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[20]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# # Parameter grid for RandomizedSearchCV
# random_param_grid = {
#     'n_estimators': np.arange(50, 200, 10),
#     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#     'max_depth': np.arange(3, 10),
#     'min_samples_split': np.arange(2, 11, 2),
#     'min_samples_leaf': np.arange(1, 5),
#     'subsample': [0.8, 0.9, 1.0],
# }

# # Initialize the classifier
# clf = GradientBoostingClassifier()

# # Randomized search on hyperparameters
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=random_param_grid,
#                                    n_iter=60, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='roc_auc')

# # Fit the random search model
# random_search.fit(X_train, y_train)

# # Best parameters from random search
# best_params_random = random_search.best_params_
# print("Best parameters found by RandomizedSearchCV:", best_params_random)


best_paras = {'subsample': 1.0, 'n_estimators': 120, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 5, 'learning_rate': 0.1}
best_model_gb = GradientBoostingClassifier(**best_paras)
best_model_gb.fit(X_train, y_train)


# Predict probabilities on the test set
y_pred_proba = best_model_gb.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC of the best model GB on the test set:", auc)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[21]:


feature_importances = best_model_gb.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Random Forest Feature Importances')
plt.show()


# In[22]:


import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Assuming X_train, y_train, X_test, y_test are already defined

# Convert data to CatBoost Pool format
train_pool = Pool(X_train, label=y_train)

# # Parameter grid for RandomizedSearchCV
# random_param_grid = {
#     'iterations': np.arange(100, 200, 10),
#     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#     'depth': np.arange(3, 10),
#     'l2_leaf_reg': [1, 3, 5, 7, 9],
#     'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'Poisson'],
# }

# # Initialize the classifier with MultiClass loss function
# clf = CatBoostClassifier(loss_function='MultiClass', cat_features=[], silent=True)
# # Randomized search on hyperparameters
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=random_param_grid,
#                                    n_iter=60, cv=5, verbose=2, random_state=42, n_jobs=-1)

# # Fit the random search model
# random_search.fit(X_train, y_train)

# # Best parameters from random search
# best_params_random = random_search.best_params_
# print("Best parameters found by RandomizedSearchCV:", best_params_random)


best_params_random = {'learning_rate': 0.1, 'l2_leaf_reg': 7, 'iterations': 130, 'depth': 8, 'bootstrap_type': 'Bernoulli'}
# Initialize the best model with the tuned parameters
best_model_cb = CatBoostClassifier(
    loss_function='MultiClass',
    cat_features=[],
    **best_params_random,
    silent=True
)
# Fit the best model on the training data
best_model_cb.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = best_model_cb.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC of the best model CatBoost on the test set:", auc)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[23]:


import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


# # Parameter grid for RandomizedSearchCV
# random_param_grid = {
#     'max_depth': np.arange(3, 10),
#     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
#     'n_estimators': np.arange(100, 200, 10),
#     'reg_alpha': [1, 3, 5, 7, 9],
#     'reg_lambda': [1, 3, 5, 7, 9],
# }

# # Initialize the classifier
# clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# # Randomized search on hyperparameters
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=random_param_grid,
#                                    n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# # Fit the random search model
# random_search.fit(X_train, y_train)

# # Best parameters from random search
# best_params_random = random_search.best_params_
# print("Best parameters found by RandomizedSearchCV:", best_params_random)


best_params_random = {'reg_lambda': 5, 'reg_alpha': 5, 'n_estimators': 140, 'max_depth': 5, 'learning_rate': 0.2}
# Initialize the best model with the tuned parameters
best_model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    **best_params_random,
    silent=True
)

# Fit the best model on the training data
best_model_xgb.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = best_model_xgb.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC of the best model XGB on the test set:", auc)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[24]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

ensemble1 = StackingClassifier(
    estimators=[('xgb', best_model_xgb), ('cb', best_model_cb), ('gb', best_model_gb)],
    final_estimator=LogisticRegression(),
    cv=5
)

ensemble1.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = ensemble1.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC of the Ensembling on the test set:", auc)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[25]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

ensemble2 = StackingClassifier(
    estimators=[('xgb', best_model_xgb), ('cb', best_model_cb), ('gb', best_model_gb), ('rf', best_model_rf)],
    final_estimator=LogisticRegression(),
    cv=5
)

ensemble2.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = ensemble2.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC of the Ensembling 4 models on the test set:", auc)

# Plotting ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[62]:


rawdatas=pd.read_csv('C:\\Users\\DELL\\Downloads\\exited\\test.csv', index_col=0)


# In[63]:


test=rawdatas.copy()


# In[64]:


test=test.drop(['CustomerId','Surname'],axis=1)


# In[65]:


test


# In[66]:


from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
test['Geography'] = le1.fit_transform(test['Geography'])

le2 = LabelEncoder()
test['Gender'] = le2.fit_transform(test['Gender'])

df.head()


# In[67]:


probas = ensemble1.predict_proba(test)[:, 1]
probas


# In[68]:


rawdatas


# In[69]:


rawdatas['Exited']=probas


# In[70]:


rawdatas


# In[71]:


rawdatas.columns.values


# In[75]:


tp2=rawdatas.drop(['Surname', 'CreditScore', 'Geography', 'Gender',
       'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary'],axis=1)


# In[76]:


tp2


# In[77]:


csv_file_path='tp2.csv'
tp2.to_csv(csv_file_path,index=False)
print("prediction saved to:",csv_file_path)


# In[ ]:




