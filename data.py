"""This class handles loading and storing the data set used to train the machine learning
algorithms and to construct featuresets from Tweets. It uses the Singleton pattern to ensure
that the data is not loaded more than once since it is a very time-consuming operation. It
also uses pickle to save the data after it has been loaded and parsed, reducing subsequent
loading times."""


import itertools
import logging
import os
import os.path
import pickle
import random
import time

from nltk import word_tokenize
from nltk import pos_tag
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
            DataSet.data_set = DataSet._load_data()
        logging.info("Returning cached data set.")
        return DataSet.data_set

    @staticmethod
    def _load_data():
        """Loads the data from all corpora. Returns a DataSet object, which contains the
        training set and the list of all words that appear in the corpora."""

        start_time = time.time()

        # Load the data set from pickle if it exists
        pickle_filepath = os.path.join('pickles', 'data.pickle')
        data = DataSet._load_data_from_pickle(pickle_filepath)
        if data is not None:
            return data
        logging.info("Loading data set from scratch...")

        # documents will be a list of tuples consisting of a body of text and its sentiment
        documents = []

        # Load the short pos/neg movie reviews (approx 10,000)
        DataSet._load_movie_reviews(documents)

        # Build list of all words that appear in data set
        word_list = DataSet._build_word_list(documents)

        # Remove useless words from word list by part of speech
        # For a list of nltk parts of speech, run nltk.help.upenn_tagset()
        allowed_pos = {'FW', 'JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'UH',
                       'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        word_list = DataSet._remove_words_by_pos(word_list, allowed_pos)

        # Build list of labeled features
        labeled_features = DataSet._build_labeled_featuresets(documents, word_list)

        # Shuffle the featuresets
        random.shuffle(labeled_features)

        # Create the DataSet object and store the data in it
        data = DataSet()
        data.training_set = labeled_features[:-100]
        data.all_features = word_list
        data.test_set = labeled_features[-100:]

        # Pickle the data set to reduce future loading times
        DataSet._save_data_to_pickle(pickle_filepath, data)

        logging.info(f"Data loading complete. Time taken: {time.time()-start_time}\n")
        return data

    @staticmethod
    def _load_movie_reviews(documents):
        """Loads reviews from the short movie review corpus. Creates tuples of
        review-sentiment pairs and appends them to documents."""

        logging.info("Loading movie reviews...")
        short_pos = unidecode(open('positive.txt').read())
        short_neg = unidecode(open('negative.txt').read())
        for review in short_pos.split('\n'):
            documents.append((review, 'pos'))
        for review in short_neg.split('\n'):
            documents.append((review, 'neg'))

    @staticmethod
    def _build_word_list(documents):
        """Returns a list of all unique, 3+ char long words in documents."""

        logging.info("Building word list...")
        all_words = [word_tokenize(document) for document, _ in documents]
        all_words = list({word.lower() for word in itertools.chain.from_iterable(all_words)
                         if len(word) > 2})
        return all_words

    @staticmethod
    def _remove_words_by_pos(word_list, allowed_pos):
        """Returns a new list of words from word_list whose part of speech is in
        allowed_pos."""

        tagged_words = pos_tag(word_list)
        return [word[0] for word in tagged_words if word[1] in allowed_pos]

    @staticmethod
    def _build_labeled_featuresets(documents, word_list):
        """Returns a list of labeled featuresets. Each labeled featureset is a tuple of a dict
        (the featureset) and a sentiment (the label). The dict contains word-bool pairs that
        indicate whether the given word from word_list appears in the corresponding
        document."""

        logging.info("Building featureset...")
        return [(DataSet.find_features(document, word_list), sentiment)
                for document, sentiment in documents]

    @staticmethod
    def find_features(document, word_features):
        """Returns a featureset of word-bool pairs. bool will be True if word from
        word_features exists in document, else False."""

        doc_words = set(word.lower() for word in word_tokenize(document))
        return {word: (word in doc_words) for word in word_features}

    @staticmethod
    def _load_data_from_pickle(pickle_filepath):
        """Attempts to load the data set from a saved pickle file. If the pickle file does not
        exist, returns None."""

        data = None
        if os.path.isfile(pickle_filepath):
            logging.info("Loading data from pickle...")
            with open(pickle_filepath, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
        return data

    @staticmethod
    def _save_data_to_pickle(pickle_filepath, data):
        """Saves the given data object to a pickle file at the given filepath."""

        logging.info("Writing pickle file...")
        if not os.path.isdir('pickles'):
            os.mkdir('pickles')
        with open(pickle_filepath, 'wb') as pickle_file:
            pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
