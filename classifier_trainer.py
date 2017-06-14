"""This class will train the machine learning classifiers with a data set."""

import nltk
import random


from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


class DataSet:
    def __init__(self):
        self.labeled_features = None
        self.all_features = None


def load_training_data():

    with open('positive_words.txt') as file:
        lines = file.readlines()
        positive_features = {line.strip(): True for line in lines}
        positive_words = [line.strip() for line in lines]

    with open('negative_words.txt') as file:
        lines = file.readlines()
        negative_features = {line.strip(): True for line in lines}
        negative_words = [line.strip() for line in lines]

    positive_features = [(positive_features, 'pos')]
    negative_features = [(negative_features, 'neg')]
    all_words = positive_words + negative_words
    training_set = positive_features + negative_features
    random.shuffle(training_set)

    data = DataSet()
    data.labeled_features = training_set
    data.all_features = all_words

    return data


def train_classifier(classifier, data_set):
    classifier.train(data_set.labeled_features)


data = load_training_data()
classy = SklearnClassifier(MultinomialNB())
train_classifier(classy, data)

text = "Today, Cirno was really bad at speedruns."
text2 = "I am happy today."
tokens = set(word_tokenize(text))
tokens2 = set(word_tokenize(text2))
test_set = {word: (word in tokens) for word in data.all_features}
test_set2 = {word: (word in tokens2) for word in data.all_features}
print(classy.classify(test_set))
print(classy.classify(test_set2))
