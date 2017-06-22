import logging
import multiprocessing

from graphing import graph
from votingclassifier import VotingClassifier
from streaming import start_tweepy

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(funcName)-30s - %(message)s')

KEYWORD = 'happy'


def main():

    classifier = VotingClassifier()
    queue = multiprocessing.Queue()

    tweepy_process = multiprocessing.Process(target=start_tweepy,
                                             args=(KEYWORD, classifier, queue))
    matplotlib_process = multiprocessing.Process(target=graph,
                                                 args=(queue,))

    tweepy_process.start()
    matplotlib_process.start()
    matplotlib_process.join()
    tweepy_process.terminate()


if __name__ == '__main__':
    main()
