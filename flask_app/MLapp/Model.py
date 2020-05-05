import numpy as np
from tensorflow.keras import applications
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import logging


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
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.backbone = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                 input_tensor=None, input_shape=(self.img_width, self.img_height, 3))
        self.model_weights_file = 'vgg16-best-weights.h5'
        self.history = None
        self.model = None

    def save_bottleneck_features(self):
        self.train_classes = self.generate_features(self.train_dir, self.n_train_samples, 'train')
        self.val_classes = self.generate_features(self.val_dir, self.n_val_samples, 'validation')
        self.test_classes = self.generate_features(self.test_dir, self.n_test_samples, 'test')

        self.n_train_samples = len(self.train_classes)
        self.n_val_samples = len(self.val_classes)
        self.n_test_samples = len(self.test_classes)

        logging.info('\nDone! Bottleneck features have been saved')

    def generate_features(self, directory, n_samples, name_str):
        logging.info('Generate ' + name_str + ' image features')
        generator = self.datagen.flow_from_directory(
            directory,
            target_size=(self.img_width, self.img_height),
            batch_size=1,
            class_mode=None,
            shuffle=False)
        features = self.backbone.predict_generator(generator, n_samples, verbose=True)
        np.save('features_' + name_str + '.npy', features)
        return generator.classes

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

        callbacks = [ModelCheckpoint(self.model_weights_file, monitor='val_acc', verbose=1, save_best_only=True)]

        self.history = model.fit(train_data, train_labels, verbose=1,
                                 epochs=self.n_epochs, batch_size=32,
                                 validation_data=(validation_data, validation_labels),
                                 callbacks=callbacks)
        self.model = self.history.model

        model_json = model.to_json()
        with open("mod_appendix.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("catvspikavskanye_VGG16_pretrained_tf_top.h5")
        logging.info("Saved model to disk")
        logging.info('Done!')

    def get_history(self):
        self.history.model.load_weights('catvspikavskanye_VGG16_pretrained_tf_top.h5')
        self.model.summary()
        acc = pd.DataFrame({'epoch': range(1, self.n_epochs + 1),
                            'training': self.history.history['accuracy'],
                            'validation': self.history.history['val_accuracy']})
        ax = acc.plot(x='epoch', figsize=(10, 6), grid=True)
        ax.set_ylabel("accuracy")
        ax.set_ylim([0., 1.0])

    def validate(self):
        validation_data = np.load('features_validation.npy')

        val_pred_class = self.model.predict_classes(validation_data, verbose=0)

        logging.info('Accuracy on validation set: ', np.mean(val_pred_class.ravel() == self.val_classes) * 100, '%')
        logging.info('\nVal loss & val_acc')
        logging.info(self.model.evaluate(validation_data, utils.to_categorical(self.val_classes), verbose=0))
