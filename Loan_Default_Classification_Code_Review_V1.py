
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')


# In[2]:


import pandas as pd
import numpy as np

from scipy import stats, integrate
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from IPython.core import display as ICD

# In[4]:


def read_loan(path, filename):
    df = pd.read_csv(path + "\\" + filename, sep='\t', header=0)
    df.columns = col_names
    '''
    Args: 
        path: Location of the import file
        filename: Name of the import file
    Returns:
        df: Imported loan level details data frame
    '''
    return df

def agg_column(df, col_name):
    '''
    Args: 
        df: Data frame the aggregation calculation is based on 
        col_name: Name of aggregated column
    Returns:
        agg_series: The return is a data series. All calcuated values <=1
    '''
    agg_series = df.groupby(col_name).loan_id.nunique() / df.loan_id.nunique()
    return agg_series

def percent_format(float_series, rename_col_index):
    '''
    Args: 
        float_series: Data series with float value
        rename_col_index: The index of column which needs 
                          to be renamed for clarification
    Returns:
        percent_df: The return is a data frame with float formated as %
    '''
    percent = float_series.mul(100).round(1).astype(str) + '%'
    percent_df = pd.DataFrame(percent).reset_index()
    percent_df.rename(columns={percent_df.columns[rename_col_index]: "percent" }, inplace=True)
    #percent_df.rename(columns = {'loan_id':'percent'}, inplace=True)  # Alt: rename a column by name
    return percent_df

# In[15]:


# Create histogram method
def histogram(df, title_name):
    '''
    Args: 
        df: Data frame or data series for plotting histogram on feature value distribution 
        title_name: Name of histogram
    Returns:
        Histogram plot using matplotlib
    '''
    return df.plot(kind='bar', title=title_name).set(xlabel='\n' + title_name, ylabel='% to Total')

# In[20]:


# Alternative way to create histogram method using feature as parameter
def histogram_alt(feature):
    df = distr_dict[feature]
    title_name = feature.split('_', 1)[0].upper()
    return df.plot(kind='bar', title=title_name).set(xlabel=title_name, ylabel='% to Total')



# In[3]:


# rename columns if needed
col_names = ['loan_id','ory','orig_upb','loan_purp','prop_type',
             'multi_unit','orig_chn','occ_stat','dti_new',
             'FICO_new', 'ltv_new', 'fhb_flag', 'no_bor', 
             'prop_type_eligible', 'MI_chl', 'dr_time_default',
             'Ever_Delinquent', 'current_status', 'claim_flag']

# In[12]:


features = ['ory', 'loan_purp', 'prop_type', 'multi_unit', 'orig_chn', 
            'occ_stat', 'dti_new', 'FICO_new', 'ltv_new', 'fhb_flag',
            'no_bor', 'prop_type_eligible', 'MI_chl', 'Ever_Delinquent',
            'claim_flag']


loan_file = "C:\\Users\\SunLix\\Data\\Project\\Default Classification","Loan_Orig_2010_2013.txt")



# In[5]:


df = read_loan(loan_file)


# In[6]:


df.head()


# In[7]:


# Get size of data
df.shape


# In[8]:


# unique number of loans
df.loan_id.nunique()


# In[9]:


# Get the list of columns along with dtypes
df.dtypes


# ### Exploratory Data Analysis 

# In[10]:


#check any column with NAN value
df.isnull().any()

# ### Features
# - **Loan Origination Year**
# - **Loan Purpose**: Purchase(P), Refinance with Cash-Out(C), Refinance Pay-off Existing Lien(N)
# - **Property Type**: Single-Family and PUD(SF), Condo and Co-Op(CO), Manufactured Housing(MH)
# - **Number of Units** 
# - **Origination Channel:** Retail(R), Broker(B), Correspondent(C)
# - **Occupancy Status:** Prim Resident or Unknown(O), Second(S), Investor(I)
# - **DTI**: Debt to Income Ratio
# - **FICO**: Borrower combined FICO score
# - **LTV**: Loan to property Value Ratio
# - **First Time Home Buyer Flag**
# - **Number of Borrowers**
# - **Property Type Eligible for MI**
# - **MI Channel**
# - **Every Deliquenty Flag**
# - **Claim Flag**


for feature in features:
    ICD.display(percent_format(agg_column(df, feature), 1))


# In[13]:


# Claim Rate by Vintage Year
(df[df['claim_flag']==1]
 .groupby('ory')
 .loan_id.nunique()/df.groupby('ory').loan_id.nunique()
).mul(100).round(1).astype(str) + '%'


# #### Distribution on Features Values

# In[14]:


# Create dictionary to map feature's distribution
distr_dict = {}
for feature in features:
    distr_series = agg_column(df, feature)
    distr_dict[feature] = distr_series



# In[16]:


histogram(distr_dict['ory'], 'Origination Year')


# In[17]:


histogram(distr_dict['loan_purp'], 'Loan Purpose')


# In[18]:


histogram(distr_dict['fhb_flag'], 'First Time Homebuyer')


# In[19]:


histogram(distr_dict['Ever_Delinquent'], 'Ever Delinquent')



# In[21]:


histogram_alt('ltv_new')


# In[22]:


histogram_alt('FICO_new')


# In[23]:


histogram_alt('dti_new')


# # Modeling

# **Due to unbalanced target data, we will need to upsample or downsample one of the class. Here I chose to downsample never deliquenty loans** 
#  - Select 10,000 per group (Ever Delinquent)

# In[24]:


target_col_name = 'Ever_Delinquent'
sample_size = 10000
df_new = df.groupby(target_col_name,as_index=False).apply(lambda x: x.sample(sample_size)).reset_index()


# In[25]:


df_new.groupby(target_col_name).loan_id.nunique()


# ### Selected Features
# - **Loan Purpose**: loan_purp 
# - **Property Type**: prop_type
# - **Occupancy Status**: occ_stat
# - **DTI**: dti_new
# - **FICO**: FICO_new
# - **LTV**: ltv_new
# - **First Time Home Buyer Indicator**: fhb_flag
# - **MI Channel**: MI_chl
# - **Number of Borrower**: no_bor
# - **Origination Year**: ory
# 
# ### Target : Ever Delinquent (Y/N)

# In[26]:


# Create a new DataFrame to just include selected features and target
selected_col_name = ['loan_purp', 'prop_type', 'occ_stat', 'dti_new',
                     'FICO_new', 'ltv_new', 'fhb_flag', 'MI_chl',
                     'no_bor','ory','Ever_Delinquent']
df_model = df_new[selected_col_name]


# In[27]:


df_model.head()


# Split table into "target" vs. "features"

# In[28]:


X = df_model.iloc[:,0:-1].values
y = df_model.iloc[:,-1].values


# **Encode Categorical Columns**
# - Use LabelEncoder to transform categorical columns to numeric value, and then use OneHotEncoder to get dummy variable
# - Create dummy variables for fields which have more than 2 values

# In[29]:


feature_ls = ['loan_purp', 'prop_type', 'occ_stat', 'dti_new', 'FICO_new', 
              'ltv_new', 'fhb_flag', 'MI_chl', 'no_bor','ory'
             ]
              
df_X = pd.DataFrame(X, columns = feature_ls)

# Create dummy variable and drop one value for each feature
df_X_dummy = pd.get_dummies(df_X, drop_first=True).astype(np.int64)


# In[30]:


df_X_dummy.shape


# In[31]:


df_X_dummy.head()


# In[32]:


# FICO related fields list - used to create model threshold
df_X_FICO = df_X_dummy[['FICO_new_650', 'FICO_new_670', 'FICO_new_690',
                        'FICO_new_710', 'FICO_new_730', 'FICO_new_750',
                        'FICO_new_770', 'FICO_new_790', 'FICO_new_800'
                       ]]

# Change the transformed dataframe to array
X = df_X_dummy.as_matrix()
X_FICO = df_X_FICO.as_matrix()


# In[33]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_FICO_train, X_FICO_test, y_FICO_train, y_FICO_test = train_test_split(X_FICO, y, test_size = 0.2, random_state = 0)


# **Creating Threshold Based on FICO Only Prediction**

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
clf_thresh = LogisticRegression(random_state = 0)
clf_thresh.fit(X_FICO_train, y_FICO_train)
y_FICO_pred = clf_thresh.predict(X_FICO_test)
print('ROC AUC Score Threshold is %s' % (roc_auc_score(y_FICO_test, y_FICO_pred)))


# In[35]:


from sklearn.model_selection import cross_val_score
accuracies_thresh = cross_val_score(estimator = clf_thresh, X = X_FICO_train, y = y_FICO_train, cv = 10)
print ('Mean of Threshold Accuracy is %s' % (accuracies_thresh.mean()))
print ('Std of Threshold Accuracy is %s' % (accuracies_thresh.std()))


# #### Applying Different Classifiers on Training Dataset

# In[36]:


from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Create a list of classifiers
classifiers = [LogisticRegression(random_state = 0),
               SVC(kernel = "linear", random_state = 0),
               SVC(kernel = "poly", random_state = 0),
               RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
               DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
               KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2),
               
              ]

classifier_name = ['LogisticReg', 'SVM', 'SVM Kernel', 'Random Forest', 'Decision Tree', 'K-NN']


# In[37]:


classifier_dict = {'LogisticReg': LogisticRegression(random_state = 0),
                   'SVM': SVC(kernel = "linear", random_state = 0),
                   'SVM Kernel': SVC(kernel = "poly", random_state = 0),
                   'Random Forest': RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
                   'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
                   'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
                  }


# In[38]:


performance_dict = {}

for clf in classifiers:
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    performance_dict[clf] = [roc_auc_score(y_test, pred)]

pd.DataFrame(performance_dict)


# In[39]:


# Alternative Way to Show roc_auc_score

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('SVMKernel', SVC(kernel = "poly")))
models.append(('RF', RandomForestClassifier()))


# In[40]:


results = []
names = []
for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, pred)
    #results.append(roc_auc)
    #names.append(name)
    #msg = "%s: %f" % (name, roc_auc_score)


"""
Interpret this part!
What is the ROC AUC score?
Which algorithms are better?
Why does the performance of some of the algorithms look similar?
Does this plot indicate other work should be done?
"""
# In[42]:


# Compare roc auc score among different models
results_auc = []
names = []
scoring_auc = 'roc_auc'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results_auc = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring_auc)
	results_auc.append(cv_results_auc)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results_auc.mean(), cv_results_auc.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm ROC AUC Score Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_auc)
ax.set_xticklabels(names)
plt.show()


# ### Find out the probablity of loan in claim with the ever deliquent status

# In[43]:


(df[df['claim_flag'] == 1]
  .groupby('Ever_Delinquent')
  .loan_id.nunique()/df.groupby('Ever_Delinquent').loan_id.nunique()
).mul(100).round(1).astype(str) + '%'

"""
Regularization should be done as part of the model
selection process above
"""
# ### Perform Regularization On Logistic Regression Classifier

# In[44]:


l1_penalty = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 60, 80, 90, 100]
regularization_dic = {}

"""
Be mindful of indentation in python. This loop isn't doing what you
what it to. It's initializing many clf objects based on the l1_penalty
then returning the last one, which you're fitting on. That's why your
regularization_dic has 1 value at the max penalty instead of 1 key
per l1_penalty.
"""
for penalty_param in l1_penalty:
    clf = LogisticRegression(penalty = 'l1', C = penalty_param)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    regularization_dic[penalty_param] = roc_auc_score(y_test, y_pred)


"""
Before sharing a notebook it's good practice to clear all the output
and re-run all the cells. That ensures some else can reproduce the work 
and will help you avoid problems caused by out-of-order code.
"""
# In[55]:


regularization_dic


# In[45]:


for penalty_param in l1_penalty:
    clf  = LogisticRegression(penalty = 'l1', C = penalty_param)
clf.fit(X_train, y_train)
roc_auc = model_selection.cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'roc_auc')
msg = "%s: %f (%f)" % (penalty_param, roc_auc.mean(), roc_auc.std())
print(msg)


# In[46]:


# Predicting the Test set results
y_pred = clf.predict(X_test)
y_pred

"""
This _looks_ like it'll be your Evaluation section. If so, call it out
as such. Here are a few recommendations on what to evaluate:
- Show the most important features and discuss them
- Plot the ROC of the training set vs. the test set
- Plot the distribution (KDE) of scores on the train vs. test set
- Plot the distribution (KDE) of scores for the "1" vs. "0" class
on the test set get a sense of how well the classifier discriminates 

Then bring it all home and evalute the actual business problem.
Based on your prediction, the value of the home, and your assumption
based on how early delinquencies translate to default - how does
your model do versus a naive estimate?
"""
# In[47]:


# Return the probablity of "1" class to training set and test set
y_hats_train = clf.predict(X_train)
delinquent_proba_train = clf.predict_proba(X_train)[:,1]
df_train = pd.DataFrame(X_train, columns = df_X_dummy.columns)
df_train.loc[:,'y_actual'] = y_train
df_train.loc[:,'y_hats'] = y_hats_train
df_train.loc[:,'delinquent_proba'] = delinquent_proba_train


# In[48]:


y_hats_test = clf.predict(X_test)
delinquent_proba_test = clf.predict_proba(X_test)[:,1]
df_test = pd.DataFrame(X_test)
df_test.loc[:,'y_actual'] = y_test
df_test.loc[:,'y_hats'] = y_hats_test
df_test.loc[:,'delinquent_proba'] = delinquent_proba_test


# In[49]:


df_train.head()


# In[50]:


df_train.groupby("y_hats").delinquent_proba.hist(alpha=0.4)


# In[51]:


df_test.groupby("y_hats").delinquent_proba.hist(alpha=0.4)


# In[52]:


df_train.groupby("y_actual").delinquent_proba.hist(alpha=0.4)


# In[53]:


df_test.groupby("y_actual").delinquent_proba.hist(alpha=0.4)


# In[54]:


import seaborn as sns
sns.kdeplot(df_test.y_actual)  

