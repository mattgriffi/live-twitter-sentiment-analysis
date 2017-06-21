import matplotlib.pyplot as plt
import tweepy
import statistics
import multiprocessing

from data import DataSet
from voting_classifier import VotingClassifier
from collections import deque


class KeywordStreamListener(tweepy.StreamListener):

    def __init__(self, classifier, queue):
        super().__init__()
        self.classifier = classifier
        self.queue = queue

    def on_status(self, status):
        text = status.text.strip()
        # if not text.startswith('RT @'):
        # print('Tweet received!')
        classification = self.classifier.classify(
            DataSet.find_features(text, DataSet.get_feature_list()))
        if self.classifier.get_most_recent_confidence() < 1.0:
            classification = 'unsure'
        self.queue.put((status.text.strip(), classification))
            # print(status.text.strip())
            # print(f"Classification: {classification}\n")

    def on_error(self, code):
        print(f"ERROR: {code}")


def main():

    keyword = 'Trump'
    classifier = VotingClassifier()
    queue = multiprocessing.Queue()

    tweepy_process = multiprocessing.Process(target=start_tweepy,
                                             args=(keyword, classifier, queue))
    matplotlib_process = multiprocessing.Process(target=start_matplotlib,
                                                 args=(queue,))

    tweepy_process.start()
    matplotlib_process.start()
    tweepy_process.join()
    matplotlib_process.join()


def start_matplotlib(queue):

    while True:
        while not queue.empty():
            print(queue.get()[0])

    # plt.figure(1)
    # plt.plot()
    # plt.show()
    # stream.disconnect()


def start_tweepy(keyword, classifier, queue):

    with open('keys.txt') as file:
        keys = [line.strip() for line in file.readlines()]

    auth = tweepy.OAuthHandler(keys[0], keys[1])
    auth.set_access_token(keys[2], keys[3])

    api = tweepy.API(auth)

    stream_listener = KeywordStreamListener(classifier, queue)
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

    stream.filter(track=[keyword])


if __name__ == '__main__':
    main()
