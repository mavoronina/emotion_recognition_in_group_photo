import cv2
import os
import argparse
import sys
import logging
from flowexecutor import FlowExecutor


def __main__():
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

    flow_executor = FlowExecutor()
    image_process(sys.argv[1], flow_executor)


def image_process(source, flow_executor):
    if not os.path.isdir(source):
        execute(cv2.imread(source), flow_executor, source)
    else:
        for filename in os.listdir(source):
            image_name = os.path.join(source, filename)
            img = cv2.imread(image_name)
            if img is not None:
                execute(img, flow_executor, image_name)


def execute(image, flow_executor, name='Result'):
    image, avg_emo = flow_executor.execute(image)
    if avg_emo == 'Neutral':
        avg_emo = 'neutral'
    elif avg_emo in ('Happy', 'Surprise'):
        avg_emo = 'positive'
    else:
        avg_emo = 'negative'
    
    if len(sys.argv) > 2:
        resdir = sys.argv[2]
    else:
        resdir = 'results'
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    new_name = '{}_{}'.format(avg_emo, os.path.basename(sys.argv[1]))
    imname = os.path.join(resdir, new_name)
    cv2.imwrite(imname, image)


if __name__ == "__main__":
    __main__()
