# Author: Íris Friðriksdóttir
# Date:
# Project: 10_boosting
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

# section1 using titanic dataset and pandas 
# get_titanic from tools, fetches the datset from kaggle, combines them and cleans
# need to have folder inside project named /data/train.csv and /data/test.csv
# need to implement better_titanic data. clean the data

# section2 using random forest, pick from sklearn.metrics

# section 3

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('c:/Users/irisf/Documents/HR-master/MachineLearning/T809DATA_2022/10_boosting/data/train.csv'):
        train = pd.read_csv('c:/Users/irisf/Documents/HR-master/MachineLearning/T809DATA_2022/10_boosting/data/train.csv')
        test = pd.read_csv('c:/Users/irisf/Documents/HR-master/MachineLearning/T809DATA_2022/10_boosting/data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)

    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)
    # Instead of dropping the Age column we replace NaN values
    # with a randomized value that is between the max value and min value.
    age_min = X_full.Age.min()
    age_max = X_full.Age.max()
    age_mean = X_full.Age.mean()
    age_std = X_full.Age.std()
    age_med = X_full.Age.median()
    sigma1 = int(age_mean-age_std)
    sigma_2 = int(age_mean+age_std)
    age_guess = random.randint(sigma1, sigma_2)
    # print(age_min)
    # print(age_max)
    # print(age_mean)
    # print(age_std)
    # print(age_med)
    # print(age_guess)
    X_full['Age'].fillna(age_guess, inplace=True)
    
    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    print(X_full['Age'])
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X


def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    # n_estimators=default value = 100, max_features = number of columns
    clf = RandomForestClassifier(n_estimators=100, max_features=15)
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    return acc, prec, rec


def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    clf = GradientBoostingClassifier()
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    return acc, prec, rec


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': [...],
        'max_depth': [...],
        'learning_rate': [...]}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    ...


def _create_submission():
    '''Create your kaggle submission
    '''
    ...
    prediction = ...
    build_kaggle_submission(prediction)



if __name__ == '__main__':
    # (tr_X, tr_y), (tst_X, tst_y), submission_X = get_titanic()
    # print(tr_X['Pclass'])
    # print(tr_X.shape)
    # print(tr_y.shape)
    # print(get_titanic())
    # print(get_better_titanic())

    # PART 1
    (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()

    # PART 2
    print(rfc_train_test(tr_X, tr_y, tst_X, tst_y))
    print(gb_train_test(tr_X, tr_y, tst_X, tst_y))