import logging
import multiprocessing

from classification import start_classify
from graphing import start_graph
from streaming import start_stream

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(funcName)-30s - %(message)s')
# logging.disable(logging.CRITICAL)

KEYWORD = 'happy'


def main():

    stream_to_classify = multiprocessing.Queue()
    classify_to_graph = multiprocessing.Queue()

    streaming_process = multiprocessing.Process(target=start_stream,
                                                args=(KEYWORD, stream_to_classify))
    classification_process = multiprocessing.Process(target=start_classify,
                                                     args=(stream_to_classify,
                                                           classify_to_graph))
    graphing_process = multiprocessing.Process(target=start_graph, args=(classify_to_graph, KEYWORD))

    streaming_process.start()
    classification_process.start()
    graphing_process.start()

    graphing_process.join()
    streaming_process.terminate()
    classification_process.terminate()


if __name__ == '__main__':
    main()
