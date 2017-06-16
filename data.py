"""This class handles loading and storing the data set used to train the machine learning algorithms
and to construct featuresets from Tweets. It uses the Singleton pattern to ensure that the data is
not loaded more than once since it is a very time-consuming operation. It also uses pickle to save
the data after it has been loaded and parsed, reducing subsequent loading times."""


import itertools
import os
import os.path
import pickle
import random
import time

from nltk import word_tokenize
from unidecode import unidecode


class DataSet:

    data_set = None

    def __init__(self):
        self.training_set = None
        self.all_features = None
        self.test_set = None

    @staticmethod
    def get_data():
        """Returns a DataSet object. Will load the data if a DataSet object has not already
        been created."""

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
        short_pos = unidecode(open('positive.txt').read())
        short_neg = unidecode(open('negative.txt').read())
        for review in short_pos.split('\n'):
            documents.append((review, 'pos'))
        for review in short_neg.split('\n'):
            documents.append((review, 'neg'))

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
        """Returns a featureset of word-bool pairs. bool will be True if word from word_features
        exists in document, else False."""

        doc_words = set(word.lower() for word in word_tokenize(document))
        return {word: (word in doc_words) for word in word_features}
