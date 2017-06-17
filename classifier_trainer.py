"""This class will train the machine learning classifiers with a data set."""

import logging
import nltk
import time

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from data import DataSet

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(funcName)-30s - %(message)s')


class ClassifierTrainer:

    trained_classifiers = []

    @staticmethod
    def get_trained_classifiers(classifier_list):
        """Returns a list of trained classifiers. classifier_list is a list of machine learning
        classifier constructor functions."""

        # Get NamedClassifiers, needed for building file paths
        named_classifiers = ClassifierTrainer._get_named_classifiers(classifier_list)

        # Check if trained classifier pickles exist and load them

        # Wrap NamedClassifiers
        ClassifierTrainer._wrap_named_classifiers(named_classifiers)

        # Train classifiers
        data = DataSet.get_data()
        ClassifierTrainer._train_classifiers(named_classifiers, data.training_set)

        # Pickle the trained classifiers to reduce future load times

        # Return list of trained classifiers
        return [classifier.classifier for classifier in named_classifiers]

    @staticmethod
    def _get_named_classifiers(classifier_list):
        """Returns a list of NamedClassifiers."""

        return [NamedClassifier(classifier) for classifier in classifier_list]

    @staticmethod
    def _wrap_named_classifiers(named_classifier_list):
        """Takes a list of NamedClassifiers and wraps each of them with SklearnClassifier."""

        for named_classifier in named_classifier_list:
            named_classifier.classifier = SklearnClassifier(named_classifier.classifier())

    @staticmethod
    def _train_classifiers(wrapped_named_classifier_list, training_set):
        """Takes a list of SklearnClassifier-wrapped NamedClassifiers and trains each of
        them."""

        for named_classifier in wrapped_named_classifier_list:
            logging.info(f"Training {named_classifier.name}...")
            start = time.time()
            named_classifier.classifier.train(training_set)
            logging.info(f"Training complete. Time taken: {time.time()-start}")


class NamedClassifier:
    """This class stores a classifier and its name."""

    def __init__(self, classifier):
        self.classifier = classifier
        self.name = classifier.__name__


def test_algorithm_accuracy(algorithm_list, testing_set):
    """Tests and prints the accuracy of each algorithm in the list."""
    for algorithm in algorithm_list:
        print(f"\n{algorithm._clf.__class__.__name__:<20} "
              f"{nltk.classify.accuracy(algorithm, testing_set)}")

        for i in range(10):
            print(f"Test {i}: {algorithm._clf.__class__.__name__} -> "
                  f"{algorithm.classify(testing_set[i][0])} : {testing_set[i][1]}")


data = DataSet.get_data()

data2 = DataSet.get_data()

data3 = DataSet.get_data()

algorithm_list = [MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier,
                  LinearSVC, NuSVC]
trained_algorithm_list = ClassifierTrainer.get_trained_classifiers(algorithm_list)
test_algorithm_accuracy(trained_algorithm_list, data.test_set)
