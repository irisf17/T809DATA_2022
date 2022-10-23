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
from sklearn.model_selection import (train_test_split, RandomizedSearchCV, GridSearchCV)
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
    # with a randomized value that is between the sigma1 value and sigma2 value.
    random.seed(1234)
    age_min = X_full.Age.min()
    age_max = X_full.Age.max()
    age_mean = X_full.Age.mean()
    age_std = X_full.Age.std()
    age_med = X_full.Age.median()
    sigma1 = int(age_mean-age_std)
    sigma_2 = int(age_mean+age_std)
    age_guess = random.randint(sigma1, sigma_2)
    # age_guess = random.randint(int(age_min), int(age_max))
    # age_guess = int(age_mean)
    
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
    # print(X_full['Age'])
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
    clf = RandomForestClassifier(random_state=42, n_estimators=100, max_features=16)
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    print("Random Forest Classifier")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    return acc, prec, rec


def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    print("Gradient Boosting Classifier")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    return acc, prec, rec


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    n_est = np.linspace(1,100,100)
    m_depth = np.linspace(1,50,50)
    l_rate = np.linspace(0.1, 1, 20)
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': n_est.astype(int),
        'max_depth': m_depth.astype(int),
        'learning_rate': l_rate.astype(float)}
    # Instantiate the regressor
    gb = GradientBoostingClassifier(random_state=42)
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=200,
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
    best_par = param_search(tr_X, tr_y)
    n_est = best_par['n_estimators']
    m_depth = best_par['max_depth']
    l_rate = best_par['learning_rate']
    clf = GradientBoostingClassifier(random_state=42, n_estimators=n_est, max_depth=m_depth, learning_rate=l_rate)
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    print("Gradient Boosting Classifier")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")


def _create_submission():
    '''Create your kaggle submission
    '''
    (X_train, y_train), (X_test, y_test), submission_X = get_better_titanic()
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    prediction = gbc.predict(submission_X)
    build_kaggle_submission(prediction)

# ------------------ INDE ---------------

def ind_feat_importance():
    (X_train, y_train), (X_test, y_test), submission_X = get_better_titanic()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    # predict = clf.predict(X_test)

    # Finding feature importance
    importance = clf.feature_importances_
     # Make a random dataset:
    height = importance
    bars = range(0, len(importance))
    y_pos = np.arange(len(bars))
    # Create bars
    plt.bar(y_pos, height)
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    # plt.title("Feature Importance")
    plt.xlabel("Feature index")
    plt.ylabel("Feature importance")
    # Show graphic
    plt.show()

def ind_cleaning_data():
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
    

    # We drop multiple columns that contain a lot of NaN values and are not important features
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],inplace=True, axis=1)

    # 'Cabin_mapped_1', 'Cabin_mapped_2', 'Cabin_mapped_3', 'Cabin_mapped_4', 
        # 'Cabin_mapped_5', 'Cabin_mapped_6', 'Cabin_mapped_7', 'Cabin_mapped_8', 'Embarked_Q', 'Embarked_S']
    # Instead of dropping the Age column we replace NaN values
    # with a randomized value that is between the sigma1 value and sigma2 value.
    
    random.seed(1234)
    age_min = X_full.Age.min()
    age_max = X_full.Age.max()
    age_mean = X_full.Age.mean()
    age_std = X_full.Age.std()
    sigma1 = int(age_mean-age_std)
    sigma_2 = int(age_mean+age_std)
    age_guess = random.randint(sigma1, sigma_2)
    X_full['Age'].fillna(age_guess, inplace=True)
    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # print(X_full['Age'])
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
    
def ind_rfc_train_test(X_train, t_train, X_test, t_test):
    # n_estimators=default value = 100, max_features = number of columns
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    print("Random Forest Classifier")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    return acc, prec, rec

def inde_param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    n_est = np.linspace(1,10,100)
    m_depth = np.linspace(1,50,50)
    l_rate = np.linspace(0.1, 1, 20)
    # Create the parameter grid
    param_grid = {
        'n_estimators': n_est.astype(int),
        'max_depth': m_depth.astype(int),
        'criterion' : ['gini', 'entropy']}
    # Instantiate the regressor
    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Fit randomized_mse to the data
    grid.fit(X, y)
    # Print the best parameters
    return grid.best_params_

def ind_rfc_train_test_best_param(X_train, t_train, X_test, t_test):
    # best para
    best_par = inde_param_search(X_train, t_train)
    n_crit = best_par['criterion']
    m_depth = best_par['max_depth']
    n_est = best_par['n_estimators']

    # n_estimators=default value = 100, max_features = number of columns
    clf = RandomForestClassifier(random_state=42, criterion=n_crit, max_depth=m_depth, n_estimators=n_est)
    clf.fit(X_train, t_train)
    guess = clf.predict(X_test)

    acc = accuracy_score(t_test, guess)
    prec = precision_score(t_test, guess)
    rec = recall_score(t_test, guess)
    print("Random Forest Classifier better parameters")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    return acc, prec, rec

def ind_create_submission():
    '''Create your kaggle submission for independent part
    '''
    (X_train, y_train), (X_test, y_test), submission_X = ind_cleaning_data()
    newX_train = X_train.drop(['Cabin_mapped_1', 'Cabin_mapped_2', 'Cabin_mapped_3', 'Cabin_mapped_4', 'Cabin_mapped_5', 'Cabin_mapped_6','Cabin_mapped_7','Cabin_mapped_8', 'Embarked_Q', 'Embarked_S'], axis=1)
    newX_test = X_test.drop(['Cabin_mapped_1', 'Cabin_mapped_2', 'Cabin_mapped_3', 'Cabin_mapped_4', 'Cabin_mapped_5', 'Cabin_mapped_6','Cabin_mapped_7','Cabin_mapped_8', 'Embarked_Q', 'Embarked_S'], axis=1)
    newSubmission_X = X_test.drop(['Cabin_mapped_1', 'Cabin_mapped_2', 'Cabin_mapped_3', 'Cabin_mapped_4', 'Cabin_mapped_5', 'Cabin_mapped_6','Cabin_mapped_7','Cabin_mapped_8', 'Embarked_Q', 'Embarked_S'], axis=1)
    # print(submission_X.shape)
    # print(newSubmission_X.shape)
    # best parameters
    best_par = inde_param_search(newX_train, y_train)
    print("Best parameters")
    print(best_par)
    n_crit = best_par['criterion']
    m_depth = best_par['max_depth']
    n_est = best_par['n_estimators']

    # n_estimators=default value = 100, max_features = number of columns
    rfc = RandomForestClassifier(criterion=n_crit, max_depth=m_depth, n_estimators=n_est)
    rfc.fit(X_train, y_train)
    predict = rfc.predict(submission_X)
    build_kaggle_submission(predict)


if __name__ == '__main__':
    '''
    # (tr_X, tr_y), (tst_X, tst_y), submission_X = get_titanic()
    # print(tr_X['Pclass'])
    # print(tr_X.shape)
    # print(tr_y.shape)
    # print(get_titanic())
    
    # PART 1    
    (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()

    # PART 2.1 random forest classification
    rfc_train_test(tr_X, tr_y, tst_X, tst_y)
    # PART 2.3 gradient boost classifier
    gb_train_test(tr_X, tr_y, tst_X, tst_y)

    
    # PART 2.5 finding the best parameters
    # best_par = param_search(tr_X, tr_y)
    # # n_est = best_par['n_estimators']
    # # m_depth = best_par['max_depth']
    # # l_rate = best_par['learning_rate']
    # print(best_par)


    # PART 2.6
    # NOT getting better values all the time, because it is a randomized search for the parameters???
    gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y)
    '''
    
    
    # PART 3
    # _create_submission()

    
    # INDE
    # Should I also turn in to kaggle my independent code if I get better acc, rec, pre..?
    # ind_feat_importance()
    # (X_train, y_train), (X_test, y_test), submission_X = ind_cleaning_data()
    '''
    newX_train = X_train.drop(['Cabin_mapped_1', 'Cabin_mapped_2', 'Cabin_mapped_3', 'Cabin_mapped_4', 'Cabin_mapped_5', 'Cabin_mapped_6','Cabin_mapped_7','Cabin_mapped_8', 'Embarked_Q', 'Embarked_S'], axis=1)
    newX_test = X_test.drop(['Cabin_mapped_1', 'Cabin_mapped_2', 'Cabin_mapped_3', 'Cabin_mapped_4', 'Cabin_mapped_5', 'Cabin_mapped_6','Cabin_mapped_7','Cabin_mapped_8', 'Embarked_Q', 'Embarked_S'], axis=1)
    # print(new_train[1:])
    print("Random Forest Classifier using all the data")
    rfc_train_test(X_train, y_train, X_test, y_test)

    print("Edited data, only important features")
    ind_rfc_train_test(newX_train, y_train, newX_test, y_test)
    # Best parameters
    # print(inde_param_search(newX_train, y_train))

    ind_rfc_train_test_best_param(newX_train, y_train, newX_test, y_test)
    '''
    # ind_create_submission()
    