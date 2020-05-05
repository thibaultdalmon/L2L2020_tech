import json

import numpy as np
from MLapp.Model import Model as Model
from MLapp.Scrapper import Scrapper as Scrapper
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)


class Argument:
    pass

model = None


# args = Argument()
#
# workspace_dir = '../'
# data_dir = workspace_dir + 'data_afo/'
# args.train_dir = data_dir + 'train/'
# args.val_dir = data_dir + 'val/'
# args.test_dir = data_dir + 'test/'
#
# args.n_train_samples = 0
# args.n_val_samples = 0
# args.n_test_samples = 0
#
# with open('parameters.json') as json_file:
#     data = json.load(json_file)
#
# args.queries = [data["query_list_kanye"],
#                 data["query_list_pikachu"],
#                 data["query_list_cat"]]
# args.labels = data["labels"]
#
# args.img_width = data["img_width"]
# args.img_height = data["img_height"]
#
# args.n_epochs = 20

@app.route("/scrape", methods=['GET', 'POST'])
def scrape():
    # user = np.array([[float(request.args.get('user'))]])
    # item = np.array([[float(request.args.get('item'))]])
    # predicted_score = trainer.predict(user, item)
    # d = {'predicted_score': predicted_score}
    # return jsonify(d)
    scrapper = Scrapper(args)
    scrapper.build_dataset()
    args.n_train_samples = scrapper.n_train_samples
    args.n_val_samples = scrapper.n_val_samples
    args.n_test_samples = scrapper.n_test_samples

@app.route("/train", methods=['GET', 'POST'])
def train():
    model = Model(args)
    model.save_bottleneck_features()
    model.train_model()


if __name__ == '__main__':
    args = Argument()

    workspace_dir = '../'
    data_dir = workspace_dir + 'data_afo/'
    args.train_dir = data_dir + 'train/'
    args.val_dir = data_dir + 'val/'
    args.test_dir = data_dir + 'test/'

    args.n_train_samples = 0
    args.n_val_samples = 0
    args.n_test_samples = 0

    with open('parameters.json') as json_file:
        data = json.load(json_file)

    args.queries = [data["query_list_kanye"],
                    data["query_list_pikachu"],
                    data["query_list_cat"]]
    args.labels = data["labels"]

    args.img_width = data["img_width"]
    args.img_height = data["img_height"]

    args.n_epochs = 20

    scrapper = Scrapper(args)
    scrapper.build_dataset()
    args.n_train_samples = scrapper.n_train_samples
    args.n_val_samples = scrapper.n_val_samples
    args.n_test_samples = scrapper.n_test_samples

    model = Model(args)
    model.save_bottleneck_features()
    model.train_model()



