import matplotlib.pyplot as plt
import time

from collections import deque
from contextlib import suppress
from wordcloud import WordCloud, STOPWORDS


WORDCLOUD_UPDATE_INTERVAL = 3


def graph(queue):

    most_recent_tweets = deque([('', 'pos')]*100 + [('', 'neg')]*100, maxlen=200)
    average_sentiments = deque([0.5]*100, maxlen=100)

    ax1, ax2 = _init_graphs()
    word_cloud_generator = _get_word_cloud_generator()

    while plt.fignum_exists(1):
        _get_tweets(queue, most_recent_tweets)
        _get_average_sentiment(most_recent_tweets, average_sentiments)
        _update_word_cloud(most_recent_tweets, word_cloud_generator, ax2)
        _update_sentiment_graph(ax1, average_sentiments)


def _get_average_sentiment(recent_tweets, recent_averages):

        recent_sentiments = [tweet[1] for tweet in recent_tweets]
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


def timer(interval):

    def decorator(func):
        last_time = 0

        def wrapper(*args, **kwargs):
            nonlocal last_time
            if time.time()-last_time > interval:
                last_time = time.time()
                func(*args, **kwargs)

        return wrapper

    return decorator


@timer(WORDCLOUD_UPDATE_INTERVAL)
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
