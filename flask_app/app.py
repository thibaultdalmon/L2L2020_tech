import json
import os

from MLapp.Model import Model as Model
from MLapp.Scraper import Scraper as Scraper
from flask import Flask, request, jsonify, flash, redirect, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)


class Argument:
    pass


args = Argument()

workspace_dir = '/usr/src/'
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

args.n_epochs = data["n_epochs"]
args.batch_size = data["batch_size"]

args.model_weights_file = data["model_weights_file"]
args.model_file = data["model_file"]


@app.route("/scrape", methods=['GET', 'POST'])
def scrape():
    # user = np.array([[float(request.args.get('user'))]])
    # item = np.array([[float(request.args.get('item'))]])
    # predicted_score = trainer.predict(user, item)
    # d = {'predicted_score': predicted_score}
    # return jsonify(d)
    scraper = Scraper(args)
    scraper.build_dataset()
    return jsonify({
        "queries": [query for queries in args.queries for query in queries]
    })


@app.route("/train", methods=['GET', 'POST'])
def train():
    args.n_train_samples = len(
        [sample for label in next(os.walk(args.train_dir))[1] for sample in next(os.walk(args.train_dir + label))[2]])
    args.n_val_samples = len(
        [sample for label in next(os.walk(args.val_dir))[1] for sample in next(os.walk(args.val_dir + label))[2]])
    args.n_test_samples = len(
        [sample for subdir in next(os.walk(args.test_dir))[1] for sample in next(os.walk(args.test_dir + subdir))[2]])

    if args.n_train_samples == 0 or args.n_test_samples == 0:
        return "Please run /train to gather data"

    model = Model(args)
    model.save_bottleneck_features()
    model.train_model()
    return jsonify({
        "model": "VGG16",
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "img_height": args.img_height,
        "img_width": args.img_width
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.isfile(args.model_file):
        return "Please /train, there are no models yet "

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(args.test_dir, filename))

        model = Model(args)
        predicted_class, score = model.predict(os.path.join(args.test_dir, filename))
        return jsonify({'predicted class': args.labels[int(predicted_class)], 'score': float(score)})


@app.route('/export', methods=['GET'])
def export():
    if not os.path.isfile(args.model_file):
        return "Please /train, there are no models yet "
    try:
        return send_file(args.model_file, as_attachment=True)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run()
