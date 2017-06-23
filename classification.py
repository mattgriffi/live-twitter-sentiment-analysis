"""This file classifies tweets that are sent to it from the streaming module. It then sends
the data to the graphing module."""

from collections import deque

from data import DataSet
from votingclassifier import VotingClassifier


MIN_ALLOWED_CONFIDENCE = 0.7


def start_classify(input_queue, output_queue):
    """Pulls tweets from input_queue, classifies them, and puts the result in output_queue."""

    # I use a maxlen deque to ensure that we don't fall too far behind the stream. If the
    # stream is sending tweets faster than they can be classified, then some tweets will
    # be lost
    tweets = deque(maxlen=20)
    classifier = VotingClassifier()

    while True:

        _get_tweets(input_queue, tweets)
        classified_tweets = _classify_tweets(classifier, tweets)
        _output_data(output_queue, classified_tweets)


def _get_tweets(input_queue, tweets_deque):
    """Takes all of the tweets in the input_queue and adds them to tweets_deque."""

    while not input_queue.empty():
        tweets_deque.append(input_queue.get())


def _classify_tweets(classifier, tweets):
    """Uses classifier to classify all tweets. Returns a list of tweets whose classification
    confidence was greater than MIN_ALLOWED_CONFIDENCE."""

    classified_tweets = []

    for tweet in tweets:
        feature_list = DataSet.get_feature_list()
        features = DataSet.find_features(tweet, feature_list)
        classification = classifier.classify(features)
        confidence = classifier.get_most_recent_confidence()

        if confidence > MIN_ALLOWED_CONFIDENCE:
            classified_tweets.append((tweet, classification))

    return classified_tweets


def _output_data(output_queue, data):
    """Puts the classified tweet tuples from data into output_queue."""

    for classified_tweet in data:
        output_queue.put(classified_tweet)
