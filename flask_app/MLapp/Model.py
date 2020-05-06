import logging

import numpy as np
from tensorflow.keras import applications
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


class Model:
    def __init__(self, args):
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.test_dir = args.test_dir
        self.img_width = args.img_width
        self.img_height = args.img_height
        self.n_train_samples = args.n_train_samples
        self.n_val_samples = args.n_val_samples
        self.n_test_samples = args.n_test_samples
        self.train_classes = None
        self.val_classes = None
        self.test_classes = None
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.backbone = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                 input_tensor=None, input_shape=(self.img_width, self.img_height, 3))
        self.model_weights_file = 'model_weights.h5'
        self.model_file = 'model.h5'
        self.history = None
        self.model = None

    def save_bottleneck_features(self):
        self.train_classes = self.generate_features(self.train_dir, self.n_train_samples, 'train')
        self.val_classes = self.generate_features(self.val_dir, self.n_val_samples, 'validation')
        self.test_classes = self.generate_features(self.test_dir, self.n_test_samples, 'test')

        self.n_train_samples = len(self.train_classes)
        self.n_val_samples = len(self.val_classes)
        self.n_test_samples = len(self.test_classes)

        logging.info('training samples ' + str(self.n_train_samples))
        logging.info('val samples ' + str(self.n_val_samples))
        logging.info('test samples ' + str(self.n_test_samples))

        logging.info('Done! Bottleneck features have been saved')

    def generate_features(self, directory, n_samples, name_str):
        logging.info('Generate ' + name_str + ' image features')
        generator = self.datagen.flow_from_directory(
            directory,
            target_size=(self.img_width, self.img_height),
            batch_size=max(1, self.batch_size),
            class_mode=None,
            shuffle=False)
        detected_samples = generator.classes
        features = self.backbone.predict_generator(generator, n_samples / max(1, self.batch_size), verbose=True)
        np.save('features_' + name_str + '.npy', features)
        return detected_samples

    def train_model(self):
        train_data = np.load('features_train.npy')
        train_labels = utils.to_categorical(self.train_classes)

        validation_data = np.load('features_validation.npy')
        validation_labels = utils.to_categorical(self.val_classes)

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [ModelCheckpoint(self.model_weights_file, monitor='val_accuracy', verbose=1, save_best_only=True)]

        self.history = model.fit(train_data, train_labels, verbose=1,
                                 epochs=self.n_epochs, batch_size=32,
                                 validation_data=(validation_data, validation_labels),
                                 callbacks=callbacks)
        self.model = model

        model.save(self.model_file)
        model.save_weights(self.model_weights_file)

        logging.info("Saved model to disk")
        logging.info('Done!')

    def validate(self):
        self.model = load_model(self.model_file)
        validation_data = np.load('features_validation.npy')
        val_pred_class = self.model.predict_classes(validation_data, verbose=0)
        logging.info('Accuracy on validation set: ', np.mean(val_pred_class.ravel() == self.val_classes) * 100, '%')
        logging.info('Val loss & val_acc')
        logging.info(self.model.evaluate(validation_data, utils.to_categorical(self.val_classes), verbose=0))

    def predict(self, filename):
        image = load_img(filename, target_size=(self.img_width, self.img_height))
        np_img = img_to_array(image)
        np_img = np_img * 1. / 255
        self.model = load_model(self.model_file)
        features = self.backbone.predict(np.expand_dims(np_img, axis=0))
        predicted_class = self.model.predict_classes(features)[0]
        score = self.model.predict_proba(features)[0][predicted_class]
        return predicted_class, score
