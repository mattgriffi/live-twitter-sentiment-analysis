import matplotlib.pyplot as plt
import time

from collections import deque
from wordcloud import WordCloud, STOPWORDS


def start_graphing(queue):

    most_recent_tweets = deque(['', 'pos']*100 + ['', 'neg']*100, maxlen=200)
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
    stop_words = set((*STOPWORDS, 'http', 'https', 'co', 'de', 'es', 'el'))
    WordC = WordCloud(background_color='white', stopwords=stop_words)
    return WordC.generate(text)
