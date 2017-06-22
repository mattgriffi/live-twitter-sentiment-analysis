"""This class instantiates and trains the machine learning classifiers. It will attempt to load
the classifiers from pickle files if possible, and will create and train any that cannot
be loaded. The newly created classifiers will be saved to pickles to reduce future loading
times. It also uses the Singleton pattern to make the list of trained classifiers available
without the risk of re-loading or re-training them."""

import logging
import os
import os.path
import pickle
import time

from nltk.classify.scikitlearn import SklearnClassifier

from data import DataSet


class ClassifierTrainer:

    trained_classifiers = []

    @staticmethod
    def get_trained_classifiers(classifier_list):
        """Returns a list of trained classifiers. classifier_list is a list of machine learning
        classifier constructor functions."""

        # If trained classifiers are already ready to go, just return them
        if ClassifierTrainer.trained_classifiers:
            logging.debug("Returning cached classifiers")
            return ClassifierTrainer._strip_names(ClassifierTrainer.trained_classifiers)

        # Get NamedClassifiers, needed for building file paths
        named_classifiers = ClassifierTrainer._get_named_classifiers(classifier_list)

        # Check if trained classifier pickles exist and load them. Will return a list of
        # classifiers that were successfully loaded, and a list of classifiers that could not
        # be loaded.
        trained_classifiers, untrained_classifiers = \
            ClassifierTrainer._load_classifier_pickles(named_classifiers)

        if untrained_classifiers:

            # Wrap NamedClassifiers with nltk's SklearnClassifier
            ClassifierTrainer._wrap_named_classifiers(untrained_classifiers)

            # Train classifiers
            data = DataSet.get_data()
            ClassifierTrainer._train_classifiers(untrained_classifiers, data.training_set)
            DataSet.unload_data()

            # Pickle the trained classifiers to reduce future load times
            ClassifierTrainer._save_classifiers_to_pickle(untrained_classifiers)

        # Save the trained classifiers for repeated access
        ClassifierTrainer.trained_classifiers = trained_classifiers + untrained_classifiers

        # Return list of trained classifiers
        return ClassifierTrainer._strip_names(ClassifierTrainer.trained_classifiers)

    @staticmethod
    def _strip_names(named_classifier_list):
        """Returns a list of classifiers (not NamedClassifiers)."""

        return [classifier.classifier for classifier in named_classifier_list]

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
            logging.debug(f"Training {named_classifier.name}")
            start = time.time()
            named_classifier.classifier.train(training_set)
            logging.debug(f"Training complete. Time taken: {time.time()-start}")

    @staticmethod
    def _load_classifier_pickles(named_classifier_list):
        """Attempts to load pre-trained classifiers from pickles. Returns two lists of
        NamedClassifiers. The first is a list of classifiers that were successfully loaded from
        pickle and are ready to go. The second is a list of classifiers that could not be
        loaded and need to be loaded manually."""

        pickle_folder = 'pickles'
        unloaded_classifiers = []
        loaded_classifiers = []
        for named_classifier in named_classifier_list:
            pickle_filepath = os.path.join(pickle_folder, named_classifier.name + '.pickle')
            if os.path.isfile(pickle_filepath):
                logging.debug(f"Loading {named_classifier.name} from pickle")
                with open(pickle_filepath, 'rb') as pickle_file:
                    named_classifier.classifier = pickle.load(pickle_file)
                loaded_classifiers.append(named_classifier)
            else:
                unloaded_classifiers.append(named_classifier)

        return loaded_classifiers, unloaded_classifiers

    @staticmethod
    def _save_classifiers_to_pickle(trained_classifiers):
        """Saves the pre-trained classifiers to pickle files."""

        if not os.path.isdir('pickles'):
            os.mkdir('pickles')
        for trained_classifier in trained_classifiers:
            pickle_filepath = os.path.join('pickles', trained_classifier.name + '.pickle')
            with open(pickle_filepath, 'wb') as pickle_file:
                logging.debug(f"Writing {trained_classifier.name} to pickle file")
                pickle.dump(trained_classifier.classifier, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)


class NamedClassifier:
    """This class stores a classifier and its name."""

    def __init__(self, classifier):
        self.classifier = classifier
        self.name = classifier.__name__
