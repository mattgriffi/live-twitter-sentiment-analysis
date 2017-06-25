"""This module handles the output of the Twitter sentiment analysis. It creates the matplotlib
graph and word cloud. It also handles processing the data after returned by the classifiers."""

import logging
import matplotlib
import matplotlib.pyplot as plt
import random
import string
import time

from collections import deque

from wordcloud import WordCloud, STOPWORDS


WORDCLOUD_UPDATE_INTERVAL = 3
MAX_TWEETS = 200
MAX_AVERAGES = 100


def start_graph(queue, keyword):
    """Creates matplotlib figure and plots, then loops and continuously updates them with the
    data pulled from the Twitter stream. This function must be called in a process separate
    from the Twitter stream."""

    # these are both fixed-size deques that only keep the most recent data
    recent_tweets, average_sentiments = _init_deques()

    sentiment_graph, word_cloud = _init_graphs()
    word_cloud_generator = _get_word_cloud_generator()

    while plt.fignum_exists(1):
        _get_tweets(queue, recent_tweets)
        _get_average_sentiment(recent_tweets, average_sentiments)
        _update_sentiment_graph(sentiment_graph, average_sentiments, keyword)
        _update_word_cloud(word_cloud, word_cloud_generator, recent_tweets)


def _init_deques():
    """Initializes the tweet deque and averages deque with placeholder values. Returns them."""

    # Fill the tweet deque half-and-half with pos and neg
    temp_data = [('', 'pos')]*(MAX_TWEETS//2) + [('', 'neg')]*(MAX_TWEETS//2)
    random.shuffle(temp_data)
    tweets = deque(temp_data, maxlen=MAX_TWEETS)
    # Fill the averages deque with 0.5's
    averages = deque([0.5]*MAX_AVERAGES, maxlen=MAX_AVERAGES)

    return tweets, averages


def timer(interval):
    """Parameterized decorator function that ensures the decorated function does not execute
    more often than every interval seconds."""

    def decorator(func):
        last_time = 0

        def wrapper(*args, **kwargs):
            nonlocal last_time
            if time.time()-last_time > interval:
                last_time = time.time()
                func(*args, **kwargs)

        return wrapper

    return decorator


def _init_graphs():
    """Creates the figure and subplots for the sentiment graph and word cloud. Enables
    interactive mode. Returns the two subplots."""

    logging.debug("Loading matplotlib graphs")
    matplotlib.rcParams['toolbar'] = 'None'
    fig = plt.figure(figsize=(12, 10))
    top = fig.add_subplot(211)
    bot = fig.add_subplot(212)
    bot.axis('off')
    plt.ion()

    return top, bot


def _get_tweets(queue, recent_tweets):
    """Pulls all of the tweets from the queue and appends them to recent_tweets."""

    while not queue.empty():
        recent_tweets.append(queue.get())


def _get_average_sentiment(recent_tweets, recent_averages):
    """Calculates the average sentiment from the tweets in recent_tweets. Appends the average
    to recent_averages."""

    recent_sentiments = [tweet[1] for tweet in recent_tweets]
    if len(recent_sentiments) > 0:
        average = recent_sentiments.count('pos') / len(recent_sentiments)
        recent_averages.append(average)


def _update_sentiment_graph(sentiment_graph, averages, keyword):
    """Redraws sentiment_graph with data from averages."""

    sentiment_graph.clear()

    sentiment_graph.plot([average for average in averages])

    sentiment_graph.axis([0, MAX_AVERAGES, 0, 1])
    sentiment_graph.set_title(f'Sentiment for keyword: {keyword}', fontdict={'fontsize': 20})
    sentiment_graph.set_ylabel('Positivity', fontdict={'fontsize': 15})
    sentiment_graph.set_yticks([x*0.25 for x in range(5)])
    sentiment_graph.set_yticklabels([f'{x}%' for x in range(0, 101, 25)])
    sentiment_graph.set_xticks([])
    sentiment_graph.grid(axis='y', alpha=0.5)
    sentiment_graph.spines['right'].set_visible(False)
    sentiment_graph.spines['top'].set_visible(False)

    # plt.draw()
    plt.pause(0.1)


@timer(WORDCLOUD_UPDATE_INTERVAL)
def _update_word_cloud(word_cloud_plot, word_cloud_generator, recent_tweets):
    """Updates word_cloud_plot using word_cloud_generator with text data from recent_tweets."""

    text = _get_text_from_tweets(recent_tweets)

    if text.strip():
        word_cloud = _get_word_cloud(text, word_cloud_generator)
        word_cloud_plot.clear()
        word_cloud_plot.imshow(word_cloud, interpolation='bilinear')
        word_cloud_plot.axis('off')


def _get_text_from_tweets(recent_tweets):
    """Concatenates all tweets in recent_tweets and returns the full text."""

    return ' '.join([tweet[0] for tweet in recent_tweets])


def _get_word_cloud(text, generator):
    """Uses generator to create a word cloud from text. Returns the word cloud."""

    return generator.generate(text)


def _get_word_cloud_generator():
    """Returns a word cloud generator."""

    stopwords = _get_stopwords()
    return WordCloud(background_color='white', stopwords=stopwords)


def _get_stopwords():
    """Returns a set of stopwords that should be ignored in the word cloud."""

    stopwords = list(STOPWORDS)

    for word in ('http', 'https', 'amp'):
        stopwords.append(word)

    # Add 1 and 2 letter permutations. This helps exclude common foreign words, shortened
    # url's, and so forth
    for c1 in string.ascii_lowercase:
        stopwords.append(c1)
        for c2 in string.ascii_lowercase:
            stopwords.append(c1 + c2)

    return set(stopwords)
