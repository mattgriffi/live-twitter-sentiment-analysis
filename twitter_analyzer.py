import matplotlib.pyplot as plt
import tweepy
import multiprocessing
import time

from data import DataSet
from voting_classifier import VotingClassifier
from collections import deque
from wordcloud import WordCloud, STOPWORDS


class KeywordStreamListener(tweepy.StreamListener):

    def __init__(self, classifier, queue):
        super().__init__()
        self.classifier = classifier
        self.queue = queue

    def on_status(self, status):
        try:
            # Try to get the full text from extended tweets
            text = status.extended_tweet['full_text'].strip()
        except AttributeError:
            # Fall back to normal text field for non-extended tweets
            text = status.text.strip()
        if not text.startswith('RT @'):
            classification = self.classifier.classify(
                DataSet.find_features(text, DataSet.get_feature_list()))
            if self.classifier.get_most_recent_confidence() < 0.5:
                classification = 'unsure'
            if classification != 'unsure':
                self.queue.put((text.strip(), classification))

    def on_error(self, code):
        print(f"ERROR: {code}")


def main():

    keyword = 'Microsoft'
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

    most_recent_tweets = deque(maxlen=200)
    average_sentiments = deque([0.5]*100, maxlen=100)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.axis('off')
    plt.ion()
    start = time.time()

    while True:

        while not queue.empty():
            most_recent_tweets.append(queue.get())

        recent_sentiments = [tweet[1] for tweet in most_recent_tweets]

        if len(recent_sentiments) > 0:

            average = recent_sentiments.count('pos') / len(recent_sentiments)
            average_sentiments.append(average)

            if time.time()-start > 10:
                start = time.time()
                text = ' '.join([tweet[0] for tweet in most_recent_tweets])
                word_cloud = get_word_cloud(text)
                ax2.clear()
                ax2.imshow(word_cloud, interpolation='bilinear')
                ax2.axis('off')

            ax1.clear()
            ax1.plot([average for average in average_sentiments])
            ax1.axis([0, 100, 0, 1])

            plt.draw()
            plt.pause(0.05)


def get_word_cloud(text):
    stop_words = set((*STOPWORDS, 'http', 'https', 'co', 'de', 'es'))
    WordC = WordCloud(background_color='white', stopwords=stop_words)
    return WordC.generate(text)


def start_tweepy(keyword, classifier, queue):

    with open('keys.txt') as file:
        keys = [line.strip() for line in file.readlines()]

    auth = tweepy.OAuthHandler(keys[0], keys[1])
    auth.set_access_token(keys[2], keys[3])

    api = tweepy.API(auth)

    stream_listener = KeywordStreamListener(classifier, queue)
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

    stream.filter(track=[keyword], stall_warnings=True)


if __name__ == '__main__':
    main()
