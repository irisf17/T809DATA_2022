# Author: Íris Friðriksdóttir
# Date: 18.okt 2022
# Project: 09_random_forests
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

# using cancer dataset, decision tree
# 1.1 
# complete the class given below

# section 2 Implementing random forest, also using the cancer data
# 2.1

# 2.2
# how important are the features in the learning

# 2.3
# out of bag error

# 2.4
# error also part of sklearn.metrics

# section 3
# barplot


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from collections import OrderedDict


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train, self.t_train)
        self.guess = self.classifier.predict(self.X_test)

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        return confusion_matrix(self.t_test, self.guess)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test, self.guess)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        return precision_score(self.t_test, self.guess)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test, self.guess)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        # tries to figure out if our selected dataset is good, if it works well
        # use import from sklearn.metrics
        return cross_val_score(self.classifier, self.X, self.t, cv=10)

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        self.importance = self.classifier.feature_importances_

        # Make a random dataset:
        height = self.importance
        bars = range(0, len(self.importance))

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

        # ind = np.argsort(list) sorts indexes from lowest to higest.
        # Then do np.flip(ind) to sort indexes from highest to lowest.
        indx = np.argsort(self.importance)
        return np.flip(indx)


def _plot_oob_error():
    RANDOM_STATE = 1337

    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels
    
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.title("OOB error")
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def _plot_extreme_oob_error():
    RANDOM_STATE = 1337

    cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels
    
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                bootstrap=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, t)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.title("Extreme OOB error")
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()



if __name__ == '__main__':
    # PART 1.1 and 1.2
    '''
    classifier_type = DecisionTreeClassifier()
    cc = CancerClassifier(classifier_type)
    print("DecisionTreeClassifier")
    print("Confusion Matrix")
    print(cc.confusion_matrix())
    print(f"Accuracy: {cc.accuracy()}")
    print(f"Precision: {cc.precision()}")
    print(f"Recall: {cc.recall()}")
    print(f"Cross validation acc: : {cc.cross_validation_accuracy()}")
    '''

    # PART 2.1 Vanilla means using all default parametes in the classifier
    # n_estimator : number of trees in the forest, default = 100
    # n_estimators=50, max_features=30
    '''
    classifier_type = RandomForestClassifier(max_features=30)
    dd = CancerClassifier(classifier_type)
    print("RandomForestClassifier")
    print("Confusion Matrix")
    print(dd.confusion_matrix())
    print(f"Accuracy: {dd.accuracy()}")
    print(f"Precision: {dd.precision()}")
    print(f"Recall: {dd.recall()}")
    print(f"Cross validation acc: : {dd.cross_validation_accuracy()}")
    '''
    # PART 2.2
    '''
    classifier_type = RandomForestClassifier()
    ee = CancerClassifier(classifier_type)
    print("feature importance")
    print(ee.feature_importance())
    '''
    # PART 2.4
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
    '''
    _plot_oob_error()
    '''

    # PART 3.1 run cancer classifier using ExtraTreesClassifier
    # Able to see all the features is readmne 08 SVM or cancer.feature_names
    '''
    classifier_type = ExtraTreesClassifier()
    ff = CancerClassifier(classifier_type)
    print("ExtraTreesClassifier")
    print("Confusion Matrix")
    print(ff.confusion_matrix())
    print(f"Accuracy: {ff.accuracy()}")
    print(f"Precision: {ff.precision()}")
    print(f"Recall: {ff.recall()}")
    print(f"Cross validation acc: : {ff.cross_validation_accuracy()}") 
    print("feature importance")
    print(ff.feature_importance())

    cancer = load_breast_cancer()
    imp_features = ff.feature_importance()
    f_names = cancer.feature_names

    print("Most important feature")
    print(f_names[imp_features[0]])
    print("Least important feature")
    print(f_names[imp_features[-1]])
    '''
    '''
    # PART 3.2
    _plot_extreme_oob_error()
    '''
    




