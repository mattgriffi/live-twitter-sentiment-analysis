import logging
import multiprocessing

from classification import start_classify
from graphing import start_graph
from streaming import start_stream

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(funcName)-30s - %(message)s')
# logging.disable(logging.CRITICAL)


def main():

    print('This program takes a keyword, then pulls tweets that contain that keyword from '
          'Twitter, passes them through a battery of machine learning classifiers to tag them '
          'as either positive or negative, then graphs the results. The results are in the '
          'form of a running-average graph and a word cloud of the most common words in the '
          'most recent tweets.')

    keyword = input('What keyword do you want to analyze? ').strip()

    stream_to_classify = multiprocessing.Queue()
    classify_to_graph = multiprocessing.Queue()

    streaming_process = multiprocessing.Process(target=start_stream,
                                                args=(keyword, stream_to_classify))

    classification_process = multiprocessing.Process(target=start_classify,
                                                     args=(stream_to_classify,
                                                           classify_to_graph))

    graphing_process = multiprocessing.Process(target=start_graph,
                                               args=(classify_to_graph, keyword))

    streaming_process.start()
    classification_process.start()
    graphing_process.start()

    graphing_process.join()
    streaming_process.terminate()
    classification_process.terminate()
    streaming_process.join()
    classification_process.join()


if __name__ == '__main__':
    main()
