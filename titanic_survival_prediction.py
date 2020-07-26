import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
gendr_res = pd.read_csv('titanic/gender_submission.csv')


train_df.shape
test_df.shape


train_df.info()

total_trn = train_df.isnull().sum().sort_values(ascending=False)
total_tst = test_df.isnull().sum().sort_values(ascending=False)

# percent of Data Missing

pct_trn = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)

pct_tst = (test_df.isnull().sum()/test_df.isnull().count()*100).sort_values(ascending=False)

desc_df = train_df.describe()

# as of now passengerId, Name, ticket column can be dropped off since it doesn't have much signifinace.

train_df_cpy = train_df.copy()
test_df_cpy = test_df.copy()

test_df_cpy =  pd.merge(test_df_cpy, gendr_res, on="PassengerId")


train_df_cpy = train_df_cpy.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df_cpy = test_df_cpy.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# Handling the missing data ???

train_df_cpy['Cabin'].unique()
# get the number of unique values
train_df_cpy['Cabin'].nunique()

dct = {'A': 1, 'B': 2, 'C': 3, 'D':4, 'E': 5,'F':6, 'G': 7, 'T': 8, 'U':9 }

data_lst = [train_df_cpy, test_df_cpy]

for dataset in data_lst:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Cabin_updtd'] = dataset['Cabin'].str[0]
    dataset['Cabin_updtd'] = dataset['Cabin_updtd'].map(dct)
    dataset['Cabin_updtd'] = dataset['Cabin_updtd'].fillna(0)
    dataset['Cabin_updtd'] = dataset['Cabin_updtd'].astype(int)
    
    
for dataset in data_lst:
    mean = dataset['Age'].mean()
    std = dataset['Age'].std()
    is_null = dataset['Age'].isnull().sum()
    
    rand_age = np.random.randint(mean-std, mean+std, size=is_null)
    
    age_updtd = dataset['Age']
    age_updtd[np.isnan(age_updtd)] =  rand_age
    dataset['Age'] =  age_updtd
    dataset['Age'] = dataset['Age'].astype(int)
    
    
train_df_cpy['Age'].isnull().sum()
train_df_cpy['Embarked'].isnull().sum()
train_df_cpy['Embarked'].value_counts()

# 'S' - has the max frequency so we assign 'S' for NAN values



for dataset in data_lst:
    dataset['Embarked'] =  dataset['Embarked'].fillna('S')

train_df_cpy['Embarked'].isnull().sum()

# Working with Data Transformation now to convert Object columns into Numerals:

train_df_cpy['Prefix'] = train_df['Name'].str.extract('([A-Za-z]+)\.')
test_df_cpy['Prefix'] = test_df['Name'].str.extract('([A-Za-z]+)\.')

train_df_cpy['Prefix'].value_counts()
test_df_cpy['Prefix'].value_counts()

prefixes = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Others':5}


for dataset in data_lst:
    dataset['Prefix'].replace(['Lady'], 'Mrs', inplace=True)
    dataset['Prefix'].replace(['Ms', 'Mme', 'Mlle', 'Countess'], 'Mrs', inplace=True)
    dataset['Prefix'].replace(['Sir', 'Rev'], 'Mr', inplace=True)
    dataset['Prefix'].replace(['Capt', 'Col', 'Dr', 'Major', 'Jonkheer', 'Don', 'Dona'], 'Others', inplace=True)
    dataset['Prefix'] = dataset['Prefix'].map(prefixes)
    
    
# converting 'Fare' column Float type to type int

for dataset in data_lst:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

gndr = {"male":0, "female":1}

for dataset in data_lst:
    dataset['Sex'] = dataset['Sex'].map(gndr)


port_embrkd = {'S': 0, 'C': 1, 'Q': 2}

for dataset in data_lst:
    dataset['Embarked'] =  dataset['Embarked'].map(port_embrkd)
    
    
for dataset in data_lst:
    dataset =  dataset.drop(['Cabin'], axis=1, inplace=True)
    
for dataset in data_lst:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age']>=12) & (dataset['Age']<=18), 'Age'] = 1
    dataset.loc[(dataset['Age']>=19) & (dataset['Age']<=22), 'Age'] = 2
    dataset.loc[(dataset['Age']> 22) & (dataset['Age']<=27), 'Age'] = 3
    dataset.loc[(dataset['Age']> 27) & (dataset['Age']<=33), 'Age'] = 4
    dataset.loc[(dataset['Age']> 33) & (dataset['Age']<=40), 'Age'] = 5
    dataset.loc[dataset['Age']>=41, 'Age'] = 6
    

for dataset in data_lst:
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset.loc[dataset['Fare'] <= 8, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] >= 32) & (dataset['Fare'] <= 99), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] >= 100) & (dataset['Fare'] <= 250), 'Fare'] = 4
    dataset.loc[dataset['Fare'] >= 251, 'Fare'] = 5
    
y_train = pd.DataFrame(train_df_cpy['Survived'])
X_train = train_df_cpy.drop('Survived', axis=1)


X_test = test_df_cpy.drop('Survived', axis=1)
y_test = pd.DataFrame(test_df_cpy['Survived'])


# Implementing Logistic Regression

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

accuracy_logreg = round(reg.score(X_train, y_train)*100, 2)
# accuracy_logreg = 82.27

results = confusion_matrix(y_test, y_pred)


# Implementing Decision Tree
from sklearn.tree import DecisionTreeClassifier

reg1 = DecisionTreeClassifier()
reg1.fit(X_train, y_train)

y_pred1 = reg1.predict(X_test)

accuracy_dcsn_tree = round(reg1.score(X_train, y_train)*100, 2)
# accuracy_dcsn_tree = 92.7
results1 = confusion_matrix(y_test, y_pred1)




# Implementing Random Forest classification

from sklearn.ensemble import RandomForestClassifier
reg2 = RandomForestClassifier(n_estimators=100)
reg2.fit(X_train, y_train)

y_pred2 = reg2.predict(X_test)
accuracy_random_forest = round(reg2.score(X_train, y_train)*100, 2)
results2 = confusion_matrix(y_test, y_pred2)

# K-fold cross validation
# K-fold cross validation randomly splits the training data into K subsets
# called folds. Random forest model would be trained and evaluated K times
# using a different fold for evalation everytime, use a different fold for testing
# while it would be trained on the remaining K-1 folds

from sklearn.model_selection import cross_val_score

rf_clf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf_clf, X_train, y_train, cv=10)
print(f"Scores: {scores}")
print(f"mean: {scores.mean()}")
print(f"standard deviation : {scores.std()}")


my_submission = pd.DataFrame({"PassengerId":test_df['PassengerId'],'Survived':y_pred2.astype(int)})
my_submission.to_csv("titanic/my_submission.csv", index=False)


my_submission['Survived'].value_counts()









































































    
    

    
    


























    
    
    


















    




    










    
    
    
    
    
    
    
    
    



































































