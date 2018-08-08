
# coding: utf-8

# In[1]:



import gc
gc.collect()
# import libraries


# In[2]:


from scipy import stats
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:


application_train = pd.read_csv("//home//mgwarada//Desktop//Ruvimbo//application_train.csv")
#application_train.head()


# In[ ]:


#colNullCnt = []
#for z in range(len(application_train.columns)):
#    colNullCnt.append([application_train.columns[z], sum(pd.isnull(application_train[application_train.columns[z]]))])
    
#colNullCnt  


# In[4]:


#application_train.shape


# In[4]:


# add debt to income ratio a key measure of capacity in credit risk management
application_train['DEBT_TO_INCOME'] = application_train['AMT_CREDIT']/ application_train['AMT_INCOME_TOTAL']
#application_train.head()


# In[5]:


bureau = pd.read_csv("/home/mgwarada/Desktop/Ruvimbo/bureau.csv")


bureau.shape


# In[19]:


bureau.info()


# In[6]:


# bureau data has duplicated customer ids, aggregating variables would help when we mere data

bureau_grouped = bureau.drop(['SK_ID_BUREAU'],axis=1).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean','max' ,'sum']).reset_index()
#bureau_grouped.head()
bureau_grouped.name= 'bureau_grouped'


# In[7]:


# to change code ASAP

def format_columns(df):
    columns = ['SK_ID_CURR']

# Iterate through the variables names
    for var in df.columns.levels[0]:
    # Skip the id name
        if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
           for stat in df.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
                columns.append(str(df.name) + " " + var + " " +str(stat))
    return columns


# In[8]:


columns = format_columns(bureau_grouped)
bureau_grouped.columns= columns 
#bureau_grouped.head()


# In[ ]:


#df for active loans for each client
#active_loans = bureau[bureau['CREDIT_ACTIVE']=='Active']
#active_loans.head()


# In[ ]:


#active_loans.shape


# In[ ]:


#active_grouped = active_loans.drop(['SK_ID_BUREAU'],axis=1).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean','max', 'sum']).reset_index()
#active_grouped.head()


# In[ ]:


#active_grouped.columns = columns 
#active_grouped.head()


# In[9]:


bureau_balance = pd.read_csv("//home/mgwarada/Desktop/Ruvimbo/bureau_balance.csv")
#bureau_balance.head()


# In[21]:


bureau_balance.info()


# In[10]:


# bureau balance max is 0 as debts are recorded as negative number , the min taken insted to represent the maximum loan advanced to a client 
bureau_balance_grouped = bureau_balance.drop(['STATUS'], axis= 1).groupby('SK_ID_BUREAU', as_index = False).agg(['count', 'mean', 'max','min', 'sum']).reset_index()
#bureau_balance_grouped.head()
bureau_balance_grouped.name= 'bureau_balance_grouped'


# In[11]:


columns = ['SK_ID_BUREAU']

# Iterate through the variables names
for var in bureau_balance_grouped.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_BUREAU':
        
        # Iterate through the stat names
        for stat in bureau_balance_grouped.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_balance_%s_%s' % (var, stat))


# In[12]:



bureau_balance_grouped.columns = columns
#bureau_balance_grouped.head()


# In[13]:


customer_id_lookup= bureau [['SK_ID_BUREAU','SK_ID_CURR']]
#customer_id_lookup.head()


# In[14]:


bureau_balance_grouped = pd.merge(bureau_balance_grouped, customer_id_lookup, how='left',left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU')
#bureau_balance_grouped.head()


# In[15]:




bureau_balance_customer = bureau_balance_grouped.drop(['SK_ID_BUREAU'],axis=1).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'sum']).reset_index()
#bureau_balance_customer.head()
bureau_balance_customer.name='bureau_balance_customer'


# In[16]:


columns= format_columns(bureau_balance_customer)
bureau_balance_customer.columns= columns
#bureau_balance_customer.head()


# In[17]:


POS_CASH_balance = pd.read_csv("//home/mgwarada/Desktop/Ruvimbo/POS_CASH_balance.csv")
#POS_CASH_balance.head()


# In[26]:


POS_CASH_balance.info()


# In[18]:


# bureau balance max is 0 as debts are recorded as negative number , the min taken insted to represent the maximum loan advanced to a client 
POS_CASH_grouped = POS_CASH_balance.drop(['SK_ID_PREV'],axis=1).groupby('SK_ID_CURR', as_index = False).agg([ 'mean', 'max', 'sum']).reset_index()
POS_CASH_grouped.name= 'POS_CASH_grouped'


# In[19]:


columns= format_columns(POS_CASH_grouped)
POS_CASH_grouped.columns= columns
#POS_CASH_grouped.head()


# In[20]:


credit_card_balance = pd.read_csv("/home/mgwarada/Desktop/Ruvimbo/credit_card_balance.csv")
#credit_card_balance.head()


# In[28]:


credit_card_balance.info()


# In[21]:


credit_card_balance_grouped = credit_card_balance.drop( ['SK_ID_PREV'], axis =1).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'sum']).reset_index()
credit_card_balance_grouped.name= 'credit_card_balance_grouped'


# In[22]:


columns = format_columns(credit_card_balance_grouped)
credit_card_balance_grouped.columns= columns 
#credit_card_balance_grouped.head()


# In[23]:


previous_application = pd.read_csv("//home/mgwarada/Desktop/Ruvimbo/previous_application.csv")
#previous_application.head()


# In[30]:


previous_application.info()


# In[24]:


previous_application_grouped = previous_application.drop(['SK_ID_PREV'],axis=1).groupby('SK_ID_CURR', as_index = False).agg([ 'mean', 'max', 'sum']).reset_index()
previous_application_grouped.name= 'previous_application_grouped'


# In[25]:


columns = format_columns(previous_application_grouped)
previous_application_grouped.columns= columns
#previous_application_grouped.head()


# In[26]:


installments_payments = pd.read_csv("/home/mgwarada/Desktop/Ruvimbo/installments_payments.csv")
#installments_payments.head()


# In[32]:


installments_payments.info()


# In[27]:



installments_payments_grouped = installments_payments.drop( ['SK_ID_PREV'],axis=1 ).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'sum']).reset_index()
installments_payments_grouped.name ='installments_payments_grouped'


# In[28]:


columns = format_columns(installments_payments_grouped)
installments_payments_grouped.columns = columns 
#installments_payments_grouped.head()


# In[29]:


#free up memory
del credit_card_balance
del POS_CASH_balance
del previous_application
del bureau_balance
del bureau
del installments_payments 

import gc
gc.collect()


# In[30]:


#merging datasets DO NOT DELETE
train_data_v0 = application_train.merge(bureau_grouped, on= 'SK_ID_CURR',how='left').merge(credit_card_balance_grouped, on= 'SK_ID_CURR',how='left').merge(installments_payments_grouped,on = 'SK_ID_CURR',how='left').merge(POS_CASH_grouped, on ='SK_ID_CURR',how='left').merge(previous_application_grouped,on ='SK_ID_CURR',how='left').merge(bureau_balance_customer, on = 'SK_ID_CURR',how='left')
train_data_v0.shape


# In[31]:


train_data_v1 =train_data_v0
#train_data_v1.head()


# In[32]:


categorical_list = ['SK_ID_CURR']
numerical_list = []
for i in train_data_v1.columns.tolist():
    if train_data_v1[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))
#numerical_list


# In[33]:


numeric_train = train_data_v1 [numerical_list]
#numeric_train.head()


# In[34]:


categorical_train = train_data_v1 [categorical_list]
#categorical_train.head()


# In[36]:


from sklearn.preprocessing import LabelEncoder


# In[37]:


le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in categorical_train:
    if categorical_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(categorical_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(categorical_train[col])
            # Transform both training and testing data
            categorical_train[col] = le.transform(categorical_train[col])
            
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# In[51]:


#create dummies for categorical data
categorical = pd.get_dummies(categorical_train.select_dtypes('object'))
categorical['SK_ID_CURR'] = categorical_train['SK_ID_CURR']
categorical.head()


# In[49]:


categorical_grouped = categorical.groupby('SK_ID_CURR',as_index = False).agg(['count', 'mean'])
categorical_grouped.name = 'categorical_grouped'
categorical_grouped.head()


# In[42]:



# List of column names
columnsc = []

# Iterate through the variables names
for var in categorical_grouped.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in categorical_grouped.columns.levels[1][:]:
            # Make a new column name for the variable and stat
            columnsc.append('categorical_grouped_%s_%s' % (var, stat))


# In[43]:


#columns = format_columns(categorical_grouped)
categorical_grouped.columns= columnsc
#categorical_grouped.head()


# In[44]:




#calculate missing values for each column
cat_percent_missing = (categorical_grouped.isnull().sum(axis = 0)/len(categorical_grouped ))*100
#round(abs(cat_percent_missing),1).sort_values(ascending=False)


# In[45]:


num_percent_missing = abs((numeric_train.isnull().sum(axis = 0)/len(numeric_train))*100)
#num_percent_missing.sort_values(ascending=False)


# In[46]:


num_percent_missing = num_percent_missing.index[num_percent_missing> 0.75]


# In[47]:


#remove variables with more than 75% of data missing
numeric_train = numeric_train.drop(columns = num_percent_missing)


# In[48]:


train_data_v1 = numeric_train.merge(categorical_grouped, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
train_data_v1.head()


# In[53]:


#free up memory
#del categorical_grouped
del bureau_grouped
del bureau_balance_grouped
del installments_payments_grouped
del credit_card_balance_grouped
del previous_application_grouped
del POS_CASH_grouped
del num_percent_missing
del cat_percent_missing
gc.collect()


# In[83]:


train= train_data_v1.drop(['SK_ID_CURR','TARGET'], axis =1)


# In[84]:


train = train.fillna(train.median())


# In[85]:


#identify multicolinearity
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = train.corr().abs()
#corr_matrix.head()


# In[86]:


#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#upper.head()


# In[87]:


threshold = 0.8

# Empty dictionary to hold correlated variables
above_threshold_vars = {}

# For each column, record the variables that are above the threshold
for col in corr_matrix:
    above_threshold_vars[col] = list(corr_matrix.index[corr_matrix[col] > threshold])


# In[88]:


# Track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# Iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))


# In[89]:


to_drop = cols_to_remove


# In[90]:


train = train.drop(columns = to_drop)


# In[74]:


#free up memory
del corr_matrix
del upper
del to_drop
gc.collect()


# In[ ]:


#train_data_v1.head()


# In[91]:


response = train_data_v1['TARGET']
feature_name = train.columns.tolist()


# In[55]:



#del numeric_train
#del categorical_train
#gc.collect()


# In[92]:


#train.fillna(train.median()).head()
train = train.fillna(train.median())


# In[77]:


def cor_selector(X, y):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


# In[93]:


#feature selection
cor_support, cor_feature = cor_selector(train,response)
print(str(len(cor_feature)), 'selected features')


# In[94]:


#from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.feature_selection import SelectFromModel


# In[82]:


#train_norm = MinMaxScaler().fit_transform(train)


# In[79]:


# wrapper method using regression see RFE: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

#rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=100, step=10, verbose=5)
#rfe_selector.fit(train_norm, response)


# In[80]:


#rfe_support = rfe_selector.get_support()
#rfe_feature = train.loc[:,rfe_support].columns.tolist()
#print(str(len(rfe_feature)), 'selected features')
#rfe_feature 


# In[ ]:


#rfe_support  = 308


# In[95]:


#embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.25*median')
#embeded_rf_selector.fit(train, response)


# In[96]:


#embeded_rf_support = embeded_rf_selector.get_support()
#embeded_rf_feature = train.loc[:,embeded_rf_support].columns.tolist()
#print(str(len(embeded_rf_feature)), 'selected features')


# In[106]:


from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
random_forest_model.fit(train, response)
features = train.columns.values


# In[155]:


rF = SelectFromModel(random_forest_model, threshold=0.005)

# Train the selector
rF.fit(train,response)


# In[156]:


features = train.columns.tolist()


# In[166]:



model_features=[]
for f_index in rF.get_support(indices=True):
    model_features.append(features[f_index])


# In[168]:



len(model_features)


# In[99]:



#original= pd.DataFrame({'Feature':feature_name[2:]})


# In[100]:


#additional_0 =  pd.DataFrame({'Pearson_corr':cor_support,
                             'Random Forest':embeded_rf_support})   
#additional_1= pd.DataFrame({'Pearson_corr':cor_support})
#additional_2= pd.DataFrame({'Logistic':rfe_support})
#original_1 = pd.concat([original,additional_0], axis =1)
#original_2 = pd.concat([original_1,additional_1], axis =1)
#original_3 =pd.concat([original_2,additional_2], axis =1)


# In[101]:


#feature_selection_df = original_1


# In[162]:


#feature_selection_df['Total']=0
#feature_selection_df['Total']=np.sum(feature_selection_df==True, axis=1)
# display the top 100
#feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
#feature_selection_df.index = range(1, len(feature_selection_df)+1)
#feature_selection_df


# In[170]:



model_features.append( 'SK_ID_CURR')
model_features.append('TARGET')
model_features


# In[171]:



train_data_final  = train_data_v1[model_features]
#del train_data_final


# In[172]:


train_response = train_data_final.TARGET
train_predictor =  train_data_final.drop(columns=['TARGET','SK_ID_CURR'],axis=1)


# In[175]:



train_predictor = train_predictor.fillna(train_predictor.median())
train_scaled = train_predictor


# In[176]:


#normalize/scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(train_scaled)
train_F_scaled = scaler.transform(train_scaled)


# In[178]:


from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter =500)
mlp.fit(train_F_scaled,train_response)


# In[179]:


#fit random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib


get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams.update({'font.size': 20})
                           
                           
randomForestModel = RandomForestClassifier(max_depth=5,random_state=0)


# In[180]:


randomForestModel.fit(train_predictor,train_response)


# In[102]:


#K-Fold cross validation
cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []
    
for (train, test), i in zip(cv.split(train_dataset,train_response), range(10)):
    randomForestModel.fit(train_dataset.iloc[train], train_response.iloc[train])
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)

plot_roc_curve(fprs, tprs);
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])


# In[ ]:


#fit chaid analysis
#from CHAID import Tree
from modshogun import PT_MULTICLASS, CHAIDTree
from numpy import array, dtype, int32

tree = Tree.


# In[ ]:


# fit neural network 


# In[89]:


# fit logistic regression 
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train_F_scaled, train_response)


# In[181]:


#KNN classifier 
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

# Train the model usinfit(X_train, y_train)g the training sets
model.fit(train_F_scaled,train_response)


# In[183]:


#naive bayes


from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()
naive_bayes_model .fit(train_predictor,train_response)


# In[185]:


y_pred = naive_bayes_model.predict(train_predictor)

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          train_predictor.shape[0],
          (train_response != y_pred).sum(),
          100*(1-(response != y_pred).sum()/train_predictor.shape[0])
))


# In[ ]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)


# In[ ]:


#stratified k fold cross validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,y): 
    print("Train:", train_index, "Validation:", val_index) 
    X_train, X_test = X[train_index], X[val_index] 
    y_train, y_test = y[train_index], y[val_index]


# In[ ]:


#ROC AUC

>>> import numpy as np
>>> from sklearn import metrics
>>> y = np.array([1, 1, 2, 2])
>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
>>> metrics.auc(fpr, tpr)
0.75

