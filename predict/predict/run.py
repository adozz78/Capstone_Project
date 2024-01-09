import json
import argparse
import os
import time
from collections import OrderedDict

from keras.models import load_model
from numpy import argsort

from preprocessing.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)


class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # TODO: CODE HERE
        # load model
        model_path = os.path.join(artefacts_path, "model.h5")
        model = load_model(model_path)

        # TODO: CODE HERE
        # load params
        params_path = os.path.join(artefacts_path, "params.json")
        with open(params_path, 'r') as params_file:
            params = json.load(params_file)

        # TODO: CODE HERE
        # load labels_to_index
        labels_index_path = os.path.join(artefacts_path, "labels_index.json")
        with open(labels_index_path, 'r') as labels_index_file:
            labels_to_index = json.load(labels_index_file)

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=5):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()


        logger.info(f"Predicting text_list=`{text_list}`")

        # TODO: CODE HERE
        # embed text_list
        embedded_texts = embed(text_list)

        # TODO: CODE HERE
        # predict tags indexes from embeddings
        predictions = self.model.predict(embedded_texts)

        # TODO: CODE HERE
        # from tags indexes compute top_k tags for each text
        predictions = [argsort(pred)[-top_k:][::-1] for pred in predictions]

        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
