import os
import json
import argparse
import time
import logging

from keras.models import Sequential
from keras.layers import Dense

from preprocessing.preprocessing.embeddings import embed
from preprocessing.preprocessing.utils import LocalTextCategorizationDataset
from preprocessing.preprocessing import utils

logger = logging.getLogger(__name__)


def train(dataset_path, train_conf, model_path, add_timestamp):
    """
    :param dataset_path: path to a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
    :param train_conf: dictionary containing training parameters, example :
            {
                batch_size: 32
                epochs: 1
                dense_dim: 64
                min_samples_per_label: 10
                verbose: 1
            }
    :param model_path: path to folder where training artefacts will be persisted
    :param add_timestamp: boolean to create artefacts in a sub folder with name equal to execution timestamp
    """

    # if add_timestamp then add sub folder with name equal to execution timestamp '%Y-%m-%d-%H-%M-%S'
    if add_timestamp:
        artefacts_path = os.path.join(model_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        artefacts_path = model_path

    # TODO: CODE HERE
    # instantiate a LocalTextCategorizationDataset, use embed method from preprocessing module for preprocess_text param
    # use train_conf for other needed params
    dataset = LocalTextCategorizationDataset(dataset_path,
                                             train_conf['batch_size'],
                                             train_ratio=0.8,
                                             min_samples_per_label=train_conf['min_samples_per_label'],
                                             preprocess_text=embed)

    logger.info(dataset)

    # TODO: CODE HERE
    # instantiate a sequential keras model
    # add a dense layer with relu activation
    # add an output layer (multiclass classification problem)
    model = Sequential()
    model.add(Dense(train_conf['dense_dim'], activation='relu', input_shape=(768,)))
    model.add(Dense(dataset.get_num_labels(), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: CODE HERE
    # model fit using data sequences
    train_history = model.fit(dataset.get_train_sequence(),
                              epochs=train_conf['epochs'],
                              batch_size=train_conf['batch_size'],
                              verbose=train_conf['verbose'])

    # scores
    scores = model.evaluate_generator(dataset.get_test_sequence(), verbose=0)

    logger.info("Test Accuracy: {:.2f}".format(scores[1] * 100))

    # TODO: CODE HERE
    # create folder artefacts_path
    os.makedirs(artefacts_path, exist_ok=True)

    # TODO: CODE HERE
    # save model in artefacts folder, name model.h5
    model.save(os.path.join(artefacts_path, "model.h5"))

    # TODO: CODE HERE
    # save train_conf used in artefacts_path/params.json
    with open(os.path.join(artefacts_path, "params.json"), 'w') as params_file:
        json.dump(train_conf, params_file)

    # TODO: CODE HERE
    # save labels index in artefacts_path/labels_index.json
    label_index_map = dataset.get_index_to_label_map()
    with open(os.path.join(artefacts_path, "labels_index.json"), 'w') as labels_index_file:
        json.dump(label_index_map, labels_index_file)

    # train_history.history is not JSON-serializable because it contains numpy arrays
    serializable_hist = {k: [float(e) for e in v] for k, v in train_history.history.items()}
    with open(os.path.join(artefacts_path, "train_output.json"), "w") as f:
        json.dump(serializable_hist, f)

    return scores[1], artefacts_path


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to training dataset")
    parser.add_argument("config_path", help="Path to Yaml file specifying training parameters")
    parser.add_argument("artefacts_path", help="Folder where training artefacts will be persisted")
    parser.add_argument("add_timestamp", action='store_true',
                        help="Create artefacts in a sub folder with name equal to execution timestamp")

    args = parser.parse_args()

    with open(args.config_path, 'r') as config_f:
        train_params = yaml.safe_load(config_f.read())

    logger.info(f"Training model with parameters: {train_params}")

    train(args.dataset_path, train_params, args.artefacts_path, args.add_timestamp)
