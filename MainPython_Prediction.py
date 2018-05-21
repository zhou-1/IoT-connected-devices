#data and feature determines limit of ML
#model and algorithm approaches this limit

import gc
import numpy as np
import pandas as pd
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')

# load data
train = pd.read_csv('Variablity Analysis.csv')
test = pd.read_csv("test.csv")
IDtest = test["Sample"]

# Show the head of the table
train.head()

# look up whole data
# print(train.info())

# outlier detection
def detect_outliers(df, n, features):
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        return multiple_outliers

        # detect outliers from gender, smoking status, BMI, age
        Outliers_to_drop = detect_outliers(train, 2, ["BMI", "Age"])

        train.loc[Outliers_to_drop] # Show the outliers rows

        # Drop outliers
        train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

# joining train set
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# check for null and missing values
# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()


# Infos
#train.info()
train.isnull().sum()
train.head()
train.dtypes
### Summarize data
# Summarie and statistics
train.describe()

# Feature analysis
####### numerical values
# Correlation matrix between numerical values (age, BMI) and variance
g = sns.heatmap(train[["Variance","BMI","Age"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#plt.show()
#age has more significative correlation with variance

# BMI feature vs variance
g = sns.factorplot(x="BMI",y="Variance",data=train,kind="bar", size = 8 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("variance probability")
#plt.show()
#23.2,23.8 and 24.4 are first three have high variance

# Age feature vs variance
g  = sns.factorplot(x="Age",y="Variance",data=train,kind="bar", size = 8 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("variance probability")
#plt.show()
#66,45,22 are first three high values


###### categorical values
# gender vs variance
g = sns.barplot(x="Gender",y="Variance",data=train)
g = g.set_ylabel("variance Probability")
#plt.show()
# Male have higher variance, not significant
train[["Gender","Variance"]].groupby('Gender').mean()
# F       0.517862
# M       0.604516

# smoking status vs variance
g = sns.factorplot(x="Smoking Status",y="Variance",data=train,kind="bar", size = 8 ,
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("variance probability")
#plt.show()
# smoking status vs variance by gender
g = sns.factorplot(x="Smoking Status", y="Variance", hue="Gender", data=train,
                   size=8, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("variance probability")
#plt.show()

# no need for filling missing values
# dataset is small, check there is all full

# convert variables in dataset all numbers
# gender
dataset["Gender"].describe()
dataset["Gender"].isnull().sum()
dataset["Gender"][dataset["Gender"].notnull()].head()
g = sns.countplot(dataset["Gender"],order=['F','M'])
dataset = pd.get_dummies(dataset, columns = ["Gender"],prefix="Gender")

# smoking status
dataset["Smoking Status"].describe()
dataset["Smoking Status"].isnull().sum()
dataset["Smoking Status"][dataset["Smoking Status"].notnull()].head()
g = sns.countplot(dataset["Smoking Status"],order=['Non-Smoker',  'Smoker', 'Ex-Smoker'])
dataset = pd.get_dummies(dataset, columns = ["Smoking Status"],prefix="Smoking Status")

# Drop Sample variable
dataset.drop(labels = ["Sample"], axis = 1, inplace = True)

#plt.show()
#print(dataset)


### Modeling
# separate train and test dataset
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Variance"],axis = 1,inplace=True)

## Separate train features and label
train["Variance"] = train["Variance"].astype(int)
Y_train = train["Variance"]
X_train = train.drop(labels = ["Variance"],axis = 1)

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=7)

# choose the SVC and AdaBoostclassifiers for the ensemble modeling.
# performed a grid search optimization for AdaBoost, and SVC classifiers

### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

######### 1. Adaboost
DTC = DecisionTreeClassifier()

#7 or 8?
adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1, 2],
              "learning_rate":  [ 0.1, 0.2, 0.3]}  #0.0001, 0.001, 0.01,    1.5

#n_jobs = 4 why not????
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

#gsadaDTC.best_score_


########### 2. ExtraTrees
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 5],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_


######## 3. RFC Parameters tunning
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 5],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_


######### 4. Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_



############### special: SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [ 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_



## feature importance of tree based classifiers
# In order to see the most informative features for the prediction
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))
names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1

#plt.show()


# Ensemble modeling
#######combining models
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=1)

votingC = votingC.fit(X_train, Y_train)

#print(votingC)



#########predict and submit results
test_Variance = pd.Series(votingC.predict(test), name="Variance")

results = pd.concat([IDtest,test_Variance],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)

