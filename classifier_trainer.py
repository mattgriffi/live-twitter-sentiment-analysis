"""This class will train the machine learning classifiers with a data set."""

import csv
import itertools
import nltk
import os
import os.path
import pickle
import random
import time
import unidecode

from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC


class DataSet:

    data_set = None

    def __init__(self):
        self.training_set = None
        self.all_features = None
        self.test_set = None

    @staticmethod
    def get_data():
        if DataSet.data_set is None:
            print("Loading data set from scratch...")
            DataSet.data_set = DataSet._load_data()
        print("Returning cached data set.")
        return DataSet.data_set

    @staticmethod
    def _load_data():
        """Loads the data from all corpora. Returns a DataSet object, which contains the training
        set and the list of all words that appear in the corpora."""

        # Load the data set from pickle if it exists
        pickle_filepath = os.path.join('pickles', 'data.pickle')
        if os.path.isfile(pickle_filepath):
            print("Loading data from pickle...")
            with open(pickle_filepath, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
            return data

        start_time = time.time()

        documents = []

        # Load the short pos/neg movie reviews (approx 10,000)
        print("Loading movie reviews...")
        short_pos = unidecode.unidecode(open('positive.txt').read())
        short_neg = unidecode.unidecode(open('negative.txt').read())
        for review in short_pos.split('\n'):
            documents.append((review, 'pos'))
        for review in short_neg.split('\n'):
            documents.append((review, 'neg'))

        # Load the pos/neg tweets from Sander's corpus (approx 900)
        print("Loading Tweets...")
        with open('sanders-twitter-0.2/full-corpus.csv', newline='\n') as file:
            reader = csv.reader(file)
            for line in reader:
                tweet = line[4][2:-1]
                sentiment = line[1][2:5]
                if sentiment in {'pos', 'neg'}:
                    documents.append((tweet, sentiment))

        # Build list of all words that appear in data set
        print("Building word list...")
        all_words = [word_tokenize(feature[0]) for feature in documents]
        all_words = list({word.lower() for word in itertools.chain.from_iterable(all_words)
                         if len(word) > 2})

        # Build list of labeled features
        print("Building featureset...")
        labeled_features = [(DataSet.find_features(feature, all_words), sentiment)
                            for feature, sentiment in documents]

        # Shuffle the featuresets
        random.shuffle(labeled_features)

        # Create the DataSet object and store the data in it
        data = DataSet()
        data.training_set = labeled_features[:10000]
        data.all_features = all_words
        data.test_set = labeled_features[10000:]

        # Pickle the data set to reduce future loading times
        if not os.path.isdir('pickles'):
            os.mkdir('pickles')
        with open(pickle_filepath, 'wb') as pickle_file:
            print("Writing pickle file...")
            pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Data loading complete. Time taken: {time.time()-start_time}\n")
        return data

    @staticmethod
    def find_features(document, word_features):
        doc_words = set(word.lower() for word in word_tokenize(document))
        return {word: (word in doc_words) for word in word_features}


def train_classifier(classifier, labeled_features):
    classifier.train(labeled_features)


def test_algorithm_accuracy(algorithm_list, testing_set):
    """Tests and prints the accuracy of each algorithm in the list."""
    for algorithm in algorithm_list:
        print(f"\n{algorithm._clf.__class__.__name__:<20} "
              f"{nltk.classify.accuracy(algorithm, testing_set)}")

        for i in range(10):
            print(f"Test {i}: {algorithm._clf.__class__.__name__} -> "
                  f"{algorithm.classify(testing_set[i][0])} : {testing_set[i][1]}")


def create_and_train_algorithms(algorithm_list, training_set):
    """Takes a list of machine learning algorithm constructor functions. Instantiates each of
    the algorithms, trains them with the given training set, and returns the list of trained
    algorithms."""
    training_list = [SklearnClassifier(algorithm()) for algorithm in algorithm_list]
    for algorithm in training_list:
        print(f"Training {algorithm._clf.__class__.__name__}...")
        start = time.time()
        algorithm.train(training_set)
        print(f"Training complete. Time taken: {time.time()-start}\n")
    return training_list


data = DataSet.get_data()

data2 = DataSet.get_data()

data3 = DataSet.get_data()

# algorithm_list = [MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier,
#                   LinearSVC, NuSVC]
# trained_algorithm_list = create_and_train_algorithms(algorithm_list, data.training_set)
# test_algorithm_accuracy(trained_algorithm_list, data.test_set)
