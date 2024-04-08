#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


rawdata=pd.read_csv('C:\\Users\\DELL\\Downloads\\rings\\train.csv')


# In[31]:


rawdata


# In[32]:


df=rawdata.copy()


# In[33]:


df.shape


# In[34]:


df.drop(['id'],axis=1,inplace=True)


# In[35]:


df.info()


# In[69]:


df['Rings'].unique()


# In[36]:


df.describe().T


# In[37]:


df.isna().sum()


# In[38]:


df.columns.values


# In[39]:


num_col=['Length', 'Diameter', 'Height', 'Whole weight',
       'Whole weight.1', 'Whole weight.2', 'Shell weight', 'Rings']


# In[40]:


num_col


# In[42]:


cat_col=['Sex']


# In[50]:


cat_col


# In[43]:


for col in num_col:
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    sns.histplot(df[col], ax=ax[0], kde=True)  
    sns.boxplot(x=df[col], ax=ax[1])
    plt.show()


# In[56]:


plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x='Sex', hue='Sex')
plt.show()


# In[58]:


from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
df['Sex'] = le1.fit_transform(df['Sex'])


df.head()


# In[59]:


plt.figure(figsize=(10,10))
sns.heatmap(df.drop("Rings", axis=1).corr(), annot = True)


# In[61]:


plt.figure(figsize=(10,10))
sns.heatmap(df.drop("Rings", axis=1).cov(), annot = True)


# In[63]:


X = df.drop("Rings", axis=1)
y = df.Rings


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[67]:


df.columns.values


# In[105]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt

# Load your data into a DataFrame (assuming it's already cleaned and preprocessed)
# Replace 'data.csv' with the path to your dataset
data = df

# Assuming 'Rings' is the target variable
X = data[['Sex', 'Length', 'Diameter', 'Height', 'Whole weight','Whole weight.1', 'Whole weight.2', 'Shell weight', 'Rings']]  # Features
y = data['Rings']  # Target variable

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_predictions = cross_val_predict(model, X_train, y_train, cv=kf)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_train, cv_predictions))
print("RMSLE:", rmsle)

# Plot RMSLE distribution
plt.figure(figsize=(8, 6))
plt.hist(cv_predictions - y_train, bins=30)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[106]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

# Assuming X and y are your features and target variable, respectively

# Define the number of folds for cross-validation
k = 5

# Initialize the KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=50)

# Initialize the model
model = LinearRegression()

# Initialize a list to store RMSLE for each fold
rmsle_scores = []

# Perform k-fold cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    predictions = model.predict(X_val)
    
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val, predictions))
    
    # Append the RMSLE to the list
    rmsle_scores.append(rmsle)

# Calculate the average RMSLE across all folds
average_rmsle = np.mean(rmsle_scores)
print("Average RMSLE:", average_rmsle)
# Plot RMSLE distribution
plt.figure(figsize=(8, 6))
plt.hist(cv_predictions - y_train, bins=30)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[108]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np
data=df
# Generate some example data
X = df.drop("Rings", axis=1)
y = df.Rings

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize individual models
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
gbm = GradientBoostingRegressor()
svm = SVR()
nn = MLPRegressor()

# Fit individual models on training data
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbm.fit(X_train, y_train)
svm.fit(X_train, y_train)
nn.fit(X_train, y_train)

# Make predictions on testing data
dt_preds = dt.predict(X_test)
rf_preds = rf.predict(X_test)
gbm_preds = gbm.predict(X_test)
svm_preds = svm.predict(X_test)
nn_preds = nn.predict(X_test)

# Combine predictions using simple averaging
ensemble_preds = (dt_preds + rf_preds + gbm_preds + svm_preds + nn_preds) / 5

# Calculate RMSLE for ensemble predictions
ensemble_rmsle = np.sqrt(mean_squared_log_error(y_test, ensemble_preds))
print("Ensemble RMSLE:", ensemble_rmsle)



# In[109]:


# Plot RMSLE distribution
plt.figure(figsize=(8, 6))
plt.hist(cv_predictions - y_train, bins=30)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[112]:





# In[113]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np
import pandas as pd

# Assuming 'df' is your DataFrame containing the data
# Define X and y
X = df.drop("Rings", axis=1)
y = df["Rings"]

# Define the number of folds for cross-validation
k = 5

# Initialize the KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=42)
cv_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize individual models
    dl = LinearRegression()
    dt = DecisionTreeRegressor()
    rf = RandomForestRegressor()
    gbm = GradientBoostingRegressor()
    svm = SVR()
    nn = MLPRegressor()

    # Fit individual models on training data
    dl.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    nn.fit(X_train, y_train)

    # Make predictions on testing data
    dl_preds = dl.predict(X_test)
    dt_preds = dt.predict(X_test)
    rf_preds = rf.predict(X_test)
    gbm_preds = gbm.predict(X_test)
    svm_preds = svm.predict(X_test)
    nn_preds = nn.predict(X_test)

    # Combine predictions using simple averaging
    ensemble_preds = (dl_preds + dt_preds + rf_preds + gbm_preds + svm_preds + nn_preds) / 6

    # Calculate MSE for ensemble predictions
    mse = mean_squared_error(y_test, ensemble_preds)
    cv_scores.append(mse)

# Calculate mean and standard deviation of MSE
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

# Print results
print("Mean CV MSE:", mean_cv_score)
print("Standard Deviation of CV MSE:", std_cv_score)


# In[114]:


# Calculate RMSLE for ensemble predictions
ensemble_rmsle = np.sqrt(mean_squared_log_error(y_test, ensemble_preds))
print("Ensemble RMSLE:", ensemble_rmsle)


# In[116]:


rawdatas=pd.read_csv('C:\\Users\\DELL\\Downloads\\rings\\test.csv')


# In[ ]:





# In[118]:


rawdatas.drop(['id'],axis=1,inplace=True)


# In[119]:


from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
rawdatas['Sex'] = le1.fit_transform(rawdatas['Sex'])


rawdatas.head()


# In[141]:


random_row=df.sample(n=1000)


# In[142]:


random_row


# In[143]:


test_df=random_row


# In[144]:


# Assuming 'test_df' is your DataFrame containing the test data
# Define X_test and y_test
X_test = test_df.drop("Rings", axis=1)
y_test = test_df["Rings"]

# Make predictions on test data using the ensemble model
ensemble_preds_test = (dl.predict(X_test) + dt.predict(X_test) + rf.predict(X_test) + 
                       gbm.predict(X_test) + svm.predict(X_test) + nn.predict(X_test)) / 6

# Calculate MSE for ensemble predictions on test data
mse_test = mean_squared_error(y_test, ensemble_preds_test)

# Print the MSE on test data
print("MSE on Test Data:", mse_test)


# In[139]:


# Concatenate actual and predicted values with the test data DataFrame
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': ensemble_preds_test})

# Optionally, you can add the features from the test data as well for reference
results_df = pd.concat([X_test, results_df], axis=1)

# Print the results DataFrame
results_df


# In[140]:


dp=results_df[['Actual','Predicted']]


# In[132]:


dp


# In[133]:


csv_file_path='dp.csv'
dp.to_csv(csv_file_path,index=False)
print("prediction saved to:",csv_file_path)


# In[148]:


# Assuming 'new_data_df' is your DataFrame containing the new data
# Define X_new (features of the new data)
X_new = rawdatas

# Make predictions on the new data using the ensemble model
ensemble_preds_new = (dl.predict(X_new) + dt.predict(X_new) + rf.predict(X_new) + 
                      gbm.predict(X_new) + svm.predict(X_new) + nn.predict(X_new)) / 6

# Print the predicted values for the new data
Predicted values for new data
ensemble_preds_new


# In[ ]:





# In[149]:


raw=pd.read_csv('C:\\Users\\DELL\\Downloads\\rings\\test.csv')


# In[158]:


raw['Rings']=ensemble_preds_new


# In[159]:


raw.head()


# In[160]:


fg=raw[['id','Rings']]


# In[161]:


fg


# In[162]:


csv_file_path='fg.csv'
fg.to_csv(csv_file_path,index=False)
print("prediction saved to:",csv_file_path)


# In[164]:


from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
raw['Sex'] = le1.fit_transform(raw['Sex'])


raw.head()


# In[165]:


plt.figure(figsize=(10,10))
sns.heatmap(raw.drop("Rings", axis=1).corr(), annot = True)


# In[172]:


df


# In[174]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot = True)


# In[ ]:




