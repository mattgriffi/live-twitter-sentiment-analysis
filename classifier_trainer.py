"""This class will train the machine learning classifiers with a data set."""

import nltk
import random


from nltk.tokenize import word_tokenize


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
    all_features = positive_features + negative_features
    random.shuffle(all_features)

    classifier = nltk.NaiveBayesClassifier.train(all_features)

    text = "Today, Cirno really pissed me off."
    tokens = set(word_tokenize(text))
    test_set = {word: (word in tokens) for word in all_words}
    print(classifier.classify(test_set))

load_training_data()

