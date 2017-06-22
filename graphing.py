import matplotlib.pyplot as plt
import time

from collections import deque
from contextlib import suppress
from wordcloud import WordCloud, STOPWORDS


def graph(queue):

    most_recent_tweets = deque(['', 'pos']*100 + ['', 'neg']*100, maxlen=200)
    average_sentiments = deque([0.5]*100, maxlen=100)

    ax1, ax2 = _init_graphs()
    word_cloud_generator = _get_word_cloud_generator

    while True:
        _get_tweets(queue, most_recent_tweets)
        _update_word_cloud(most_recent_tweets, word_cloud_generator, ax2)
        _update_sentiment_graph(ax1, average_sentiments)


def _get_average_sentiment(recent_averages):

        recent_sentiments = [tweet[1] for tweet in recent_averages]
        with suppress(ZeroDivisionError):
            average = recent_sentiments.count('pos') / len(recent_sentiments)
            recent_averages.append(average)


def _get_tweets(queue, recent_tweets):

    while not queue.empty():
        recent_tweets.append(queue.get())


def _init_graphs():

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.axis('off')
    plt.ion()

    return ax1, ax2


def _get_text_from_tweets(recent_tweets):

    return ' '.join([tweet[0] for tweet in recent_tweets])


def _update_sentiment_graph(sentiment_graph, averages):

    sentiment_graph.clear()
    sentiment_graph.plot([average for average in averages])
    sentiment_graph.axis([0, 100, 0, 1])

    plt.draw()
    plt.pause(0.05)


def timer(func):

    if Timer.time is None or time.time()-Timer.time > 5:
        Timer.time = time.time()

        def wrapper():
            func()

        return wrapper


@timer
def _update_word_cloud(recent_tweets, word_cloud_generator, word_cloud_plot):

    text = _get_text_from_tweets(recent_tweets)

    word_cloud = _get_word_cloud(text, word_cloud_generator)
    word_cloud_plot.clear()
    word_cloud_plot.imshow(word_cloud, interpolation='bilinear')
    word_cloud_plot.axis('off')


def _get_word_cloud_generator():

    stop_words = set((*STOPWORDS, 'http', 'https', 'co', 'de', 'es', 'el'))
    return WordCloud(background_color='white', stopwords=stop_words)


def _get_word_cloud(text, generator):

    return generator.generate(text)


class Timer:
    time = None
