"""Train the model"""
import argparse
import logging
import os
from easydict import EasyDict
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

from model.input_fn import input_fn
from model.utils import get_config_from_json
from model.utils import set_logger
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = get_config_from_json(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    img_dir = os.path.join(data_dir, "train")
    label_dir = os.path.join(data_dir, "train_masks")

    df_train = pd.read_csv(os.path.join(data_dir, 'train_masks.csv'))
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    x_train_filenames = []
    y_train_filenames = []
    for img_id in ids_train:
        x_train_filenames.append(os.path.join(img_dir, "{}.jpg".format(img_id)))
        y_train_filenames.append(os.path.join(label_dir, "{}_mask.gif".format(img_id)))

    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
                    train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=42)

    params['train_size'] = len(x_train_filenames)
    # Create the two iterators over the two datasets
    train_inputs = input_fn(x_train_filenames, y_train_filenames, params)

    params_eval = EasyDict(params)
    params_eval.horizontal_flip = False
    params_eval.width_shift_range = 0.0
    params_eval.height_shift_range = 0.0
    params_eval.shuffle = False
    params_eval['eval_size'] = len(x_val_filenames)
    params['eval_size'] = len(x_val_filenames)      # used for evaluation
    eval_inputs = input_fn(x_val_filenames, y_val_filenames, params_eval)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params_eval, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
