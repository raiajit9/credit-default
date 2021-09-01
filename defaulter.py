#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Default Prediction

# In[1]:


#Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### 1. Read Data

# In[2]:


# Loading dataset
df = pd.read_csv("C:/Users/WIN/Downloads/UCI_Credit_Card.csv")
df.head()


# #### 2. Exploration Data Analysis

# In[3]:


df.shape


# In[4]:


sns.countplot(df['default.payment.next.month'])
plt.show()


# In[5]:


print("No. of Default Payment Next Month (YES):", np.count_nonzero(df['default.payment.next.month']))


# In[6]:


print("No. of Default Payment Next Month (NO):",df[df['default.payment.next.month']==0].shape[0])


# In[7]:


df.count()


# In[8]:


df.describe()


# In[9]:


plt.figure(figsize=(20,10))
sns.pairplot(data=df, hue='default.payment.next.month', vars=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3'])


# In[10]:


plt.subplots(figsize=(20,5))
plt.subplot(121)
sns.distplot(df.LIMIT_BAL)

plt.subplot(122)
sns.distplot(df.AGE)

plt.show()


# ###### - We have more number of clients having limiting balance between 0 to 200000 currency.
# ###### - We have more number of clients from age bracket of 20 to 45, i.e., customer from mostly young to mid aged groups.

# In[11]:


columns = ['SEX', 'MARRIAGE']
for i in columns:
    plot=sns.FacetGrid(df, row='default.payment.next.month', col=i)
    plot.map(plt.hist,'AGE')
    plt.show


# ##### - We can observe that females of age group 20-30 have very high tendency to default payment compared to males in all age brackets.
# 
# ##### - We can observe that married people between age bracket of 30 and 50 and unmarried customer of age 20-30 tend to default payment with unmarried customer higher probability to default payment

# In[12]:


pay = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
bill = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
col = ['r','y','b','g','black','orange']
for i,j,k in zip(pay,bill,col):
    plt.scatter(data=df,x=i, y=j, c=k, s=1)
    plt.xlabel("Payment in past 6 months", fontsize=10)
    plt.ylabel("Bill amount in past 6 months", fontsize=10)
    plt.show()
    


# ##### -Above plot indicates that there is higher proportion of customer for whom the bill amount is high but payment done against the same is very low. This we can infer since maximum number of datapoints are closely packed along the Y-axis near to 0 on X-axis

# In[13]:


plt.subplots(figsize=(25,20))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[14]:


# creating second dataframe by dropping target
df_1 = df.drop(['default.payment.next.month'], axis=1)
plt.figure(figsize=(15,5))
ax = sns.barplot(df_1.corrwith(df['default.payment.next.month']).index, df_1.corrwith(df['default.payment.next.month']))
ax.tick_params(labelrotation=90)


# ##### -We can observed  that next month default prediction is dependent on repayment status of past six months of all the features given to us. But there is multicollinearity between the Repayment Status features.

# ##### Splitting Data into Training & Testing

# In[15]:


X = df.drop(['default.payment.next.month','ID'], axis=1)
y = df['default.payment.next.month']


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print('Shape of X_train:', X_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_test:', y_test.shape)


# In[17]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# <h3 align="center"> Building Classification Algorithm Model

# #### i) Linear regression

# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
lr_score_train = print('Train accuracy of Logistic Regression:',lr.score(X_train,y_train))
lr_score_test = print('Test accuracy of Logistic Regression:',lr.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,lr_pred))
print('****'*20)
lr_auc = roc_auc_score(y_test,lr_pred)
print('Roc_Auc_score :',lr_auc)
sns.heatmap(confusion_matrix(y_test,lr_pred), annot=True)
print('****'*20)


# #### ii) K Nearest kneighbours

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
knn_score_train = print('Train accuracy of KNeighborsClassifier:',knn.score(X_train,y_train))
knn_score_test = print('Test accuracy of KNeighborsClassifier:',knn.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,knn_pred))
print('****'*20)
knn_auc = roc_auc_score(y_test,knn_pred)
print('Roc_Auc_score :',knn_auc)
sns.heatmap(confusion_matrix(y_test,knn_pred), annot=True)
print('****'*20)


# #### iii) Naive bayes

# In[20]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_pred = nb.predict(X_test)
nb_score_train = print('Train accuracy of Naive_bayes:',nb.score(X_train,y_train))
nb_score_test = print('Test accuracy of Naive_bayes:',nb.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,nb_pred))
print('****'*20)
nb_auc = roc_auc_score(y_test,nb_pred)
print('Roc_Auc_score :',nb_auc)
sns.heatmap(confusion_matrix(y_test,nb_pred), annot=True)
print('****'*20)


# #### iv) Support Vector Machine

# In[21]:


from sklearn.svm import SVC
sv = SVC(kernel='rbf')
sv.fit(X_train,y_train)
sv_pred = sv.predict(X_test)
sv_score_train = print('Train accuracy of SVM:',sv.score(X_train,y_train))
sv_score_test = print('Test accuracy of SVM:',sv.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,sv_pred))
print('****'*20)
sv_auc = roc_auc_score(y_test,sv_pred)
print('Roc_Auc_score :',sv_auc)
sns.heatmap(confusion_matrix(y_test,sv_pred), annot=True)
print('****'*20)


# #### V) Decision Tree 

# In[22]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_pred = dt.predict(X_test)
dt_score_train = print('Train accuracy of Decision Tree:',dt.score(X_train,y_train))
dt_score_test = print('Test accuracy of Decision Tree:',dt.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,dt_pred))
print('****'*20)
dt_auc = roc_auc_score(y_test,dt_pred)
print('Roc_Auc_score :',dt_auc)
sns.heatmap(confusion_matrix(y_test,dt_pred), annot=True)
print('****'*20)


# from sklearn.ensemble import RandomForestClassifier
# RF = RandomForestClassifier()
# from sklearn.model_selection import GridSearchCV
# n_estimators = [100, 300, 500, 800, 1200]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 
# 
# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)
# 
# gridF = GridSearchCV(RF, hyperF, cv = 3, verbose = 1, n_jobs = -1)
# bestF = gridF.fit(X_train, y_train)

# #### VI) Random Forest

# In[23]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train,y_train)
RF_pred = RF.predict(X_test)
RF_score_train = print('Train accuracy of Random Forest:',RF.score(X_train,y_train))
RF_score_test = print('Test accuracy of Random Forest:',RF.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,RF_pred))
print('****'*20)
RF_auc = roc_auc_score(y_test,RF_pred)
print('Roc_Auc_score :',RF_auc)
sns.heatmap(confusion_matrix(y_test,RF_pred), annot=True)
print('****'*20)


# #### Vii) XGBoost

# In[24]:


import xgboost
from sklearn.model_selection import RandomizedSearchCV
classifier = xgboost.XGBClassifier()


# In[26]:


# hyper parameter optimization
params = {
     "learning_rate"    : [0.05,0.10,0.15,0.20,0.25,0.30],
     "max_depth"        : [3,4,5,6,8,10,12,15],
     "min_child_weight" : [1,3,5,7],
     "gamma"             : [0.1,0.2,0.3,0.4],
     "colsample_bytree" : [0.3,0.4,0.5,0.7]
 }


# In[27]:


random_search = RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[28]:


random_search.fit(X_train,y_train.ravel())


# In[29]:


random_search.best_estimator_


# In[31]:


xgb = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.15, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
xgb_score_train = print('Train accuracy of Xg boost:',xgb.score(X_train,y_train))
xgb_score_test = print('Test accuracy of Xg boost:',xgb.score(X_test,y_test))
print('****'*20)
print("Classification_report:\n",classification_report(y_test,xgb_pred))
print('****'*20)
xgb_auc = roc_auc_score(y_test,xgb_pred)
print('Roc_Auc_score :',xgb_auc)
sns.heatmap(confusion_matrix(y_test,xgb_pred), annot=True)
print('****'*20)


# <h3 align="center"> Comparision of All Classifier ROC_AUC Curve

# In[34]:


from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve


# In[36]:


model = [lr,knn,nb,sv,dt,RF,xgb]
for i in model:
    plot_roc_curve(i,X_test,y_test)


# In[37]:


data = {'Model Accuracy' : [81.13,7923.,71.88,81.10,73.06,81.86,82.15],
    'Model AUC'      : [0.73,0.71,0.74,0.73,0.62,0.77,0.79]}
result = pd.DataFrame(data, index = ['Logistic','KNN','Naive bayes','SVM','Decision tree','Random Forest','XG boost' ])


# In[38]:


result


# In[39]:


import pickle


# In[40]:


with open('model_pickle', 'wb') as f:
    pickle.dump(xgb,f)


# In[42]:


with open('model_pickle','rb') as f:
    m = pickle.load(f)


# In[56]:


m.predict(X_test)

