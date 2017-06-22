"""This classifier uses a list of machine learning classification algorithms. It is used to
classify a text, then return a majority vote of the classifications determined by the multiple
classifiers."""

import statistics

from nltk import ClassifierI
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from classifier_trainer import ClassifierTrainer


class VotingClassifier(ClassifierI):

    def __init__(self):
        self.classifiers = self._get_classifiers()
        self.confidence = None

    def _get_classifiers(self):
        classifier_list = [MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier,
                           LinearSVC, DecisionTreeClassifier, MLPClassifier]
        return ClassifierTrainer.get_trained_classifiers(classifier_list)

    def classify(self, featureset):
        results = []
        for classifier in self.classifiers:
            results.append(classifier.classify(featureset))
        mode = statistics.mode(results)
        self.confidence = results.count(mode) / len(results)
        return mode

    def get_most_recent_confidence(self):
        return self.confidence
