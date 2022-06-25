from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import os


CLASSES = 'classes'
FORMAT = 'format'
EPOCHS = 'epochs'
IMAGE_SIZE = 'img_size'
DATA = 'data'
TEST_DATA = 't_data'
TRAIN_DATA = 'train_data'
HISTORY = 'history'
LABELS = 'labels'
TEST_LABELS = 't_labels'
CLASS_NAMES = 'c_names'
BATCH_SIZE = 'b_size'

HISTORY_FILE = 'h_file'
TEST_SIZE = 't_size'
SEED = 'seed'
MODEL_NAME = 'm_name'
DATA_PATH = 'd_path'
MODEL = 'model'

program_parameters = {}
model_parameters = {}


def init_program_parameters():
    program_parameters[DATA_PATH] = "data/"
    program_parameters[TEST_DATA] = program_parameters[DATA_PATH] + "Test.csv"
    program_parameters[TRAIN_DATA] = program_parameters[DATA_PATH] + "Train"
    program_parameters[CLASSES] = 43
    program_parameters[CLASS_NAMES] = \
        {
            1:'Speed limit (20km/h)', 2:'Speed limit (30km/h)', 3:'Speed limit (50km/h)', 4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)', 6:'Speed limit (80km/h)', 7:'End of speed limit (80km/h)', 8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)', 10:'No passing', 11:'No passing veh over 3.5 tons', 12:'Right-of-way at intersection',
            13:'Priority road', 14:'Yield', 15:'Stop', 16:'No vehicles', 17:'Veh > 3.5 tons prohibited', 18:'No entry',
            19:'General caution', 20:'Dangerous curve left', 21:'Dangerous curve right', 22:'Double curve',
            23:'Bumpy road', 24:'Slippery road', 25:'Road narrows on the right', 26:'Road work', 27:'Traffic signals',
            28:'Pedestrians', 29:'Children crossing', 30:'Bicycles crossing', 31:'Beware of ice/snow', 32:'Wild animals crossing',
            33:'End speed + passing limits', 34:'Turn right ahead', 35:'Turn left ahead', 36:'Ahead only', 37:'Go straight or right',
            38:'Go straight or left', 39:'Keep right', 40:'Keep left', 41:'Roundabout mandatory', 42:'End of no passing',
            43:'End no passing veh > 3.5 tons'
        }

    model_parameters[HISTORY_FILE] = 'ModelHistory.npy'
    model_parameters[MODEL_NAME] = "TrafficDetection"
    model_parameters[IMAGE_SIZE] = (30, 30)
    model_parameters[TEST_SIZE] = 0.3
    model_parameters[BATCH_SIZE] = 32
    model_parameters[EPOCHS] = 15
    model_parameters[SEED] = 42


def parse_data():
    cur_path = os.getcwd()
    labels = []
    data = []

    for i in range(program_parameters[CLASSES]):
        path = os.path.join(cur_path, program_parameters[TRAIN_DATA], str(i))
        images = os.listdir(path)
        for image in images:
            try:
                image = Image.open(path + '/' + image)
                image = image.resize(model_parameters[IMAGE_SIZE])
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    labels = np.array(labels)
    data = np.array(data)

    return data, labels
    

def create_model(data, labels):
    x_train, x_test, y_train, y_test = \
        train_test_split(data, labels, test_size=model_parameters[TEST_SIZE], random_state=model_parameters[SEED])

    y_train = to_categorical(y_train, program_parameters[CLASSES])
    y_test = to_categorical(y_test, program_parameters[CLASSES])

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(program_parameters[CLASSES], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    fit_history = model.fit(x_train, y_train, batch_size=model_parameters[BATCH_SIZE], epochs=model_parameters[EPOCHS], validation_data=(x_test, y_test))

    # Saving trained model
    model.save(model_parameters[MODEL_NAME])
    # Saving train history
    np.save(model_parameters[HISTORY_FILE], fit_history.history)

    return model, fit_history


def load_model_history():
    from keras.models import load_model
    model = load_model(model_parameters[MODEL_NAME])
    history = np.load(model_parameters[HISTORY_FILE], allow_pickle=True).item()

    return model, history


def visualize(history):
    plt.figure(0)
    plt.plot(history['accuracy'], label='training accuracy')
    plt.plot(history['val_accuracy'], label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(1, model_parameters[EPOCHS] + 1, 1))
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.show()
    plt.figure(1)
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(np.arange(1, model_parameters[EPOCHS] + 1, 1))
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.show()


def test_data(model):
    # testing accuracy on test dataset
    cur_path = os.getcwd()
    path = os.path.join(cur_path, program_parameters[TEST_DATA])
    y_test = pd.read_csv(path)
    labels = y_test["ClassId"].values
    images = y_test["Path"].values
    data = []

    for image in images:
        image = Image.open(program_parameters[DATA_PATH] + image)
        image = image.resize(model_parameters[IMAGE_SIZE])
        data.append(np.array(image))
    x_test = np.array(data)
    pred = model.predict(x_test)
    predicted_labels = np.argmax(pred, axis=1)

    # Accuracy with the test data
    #labels = to_categorical(labels, CLASSES)

    print(f"Accuracy for {len(x_test)} cases: {int(accuracy_score(labels, predicted_labels) * 100)}%")


def predict_case(model, image_path, label='No prediction'):
    cur_path = os.getcwd()
    path = os.path.join(cur_path, image_path)

    image = Image.open(path)
    image = image.resize(model_parameters[IMAGE_SIZE])

    x_test = np.array([np.array(image)[:, :, :3]])
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    print(f'Prediction for image ({image_path}) is: [{program_parameters[CLASS_NAMES][predicted_labels[0]]}]. Initial: {label}')


def parse_arguments():
    cur_path = os.getcwd()
    parser = argparse.ArgumentParser(description='This script determine traffic signs. Args:'
                                                 '\n-i for input image to be predicted'
                                                 '\n-c to create new model'
                                                 '\n-t to test model')
    parser.add_argument('-i',
            '--image',
            type=str,
            required=True,
            help=f'An image file that will be processed. For instance ({cur_path}/data/Test/00000.png).')
    parser.add_argument('-c', '--create', action=argparse.BooleanOptionalAction, help='Creates new model')
    parser.add_argument('-t', '--test', action=argparse.BooleanOptionalAction, help='Tests the model')
    return parser.parse_args()

if __name__ == '__main__':
    print('--- Parsing arguments ---')
    arguments = parse_arguments()
    print('--- Done')

    # Program data initialization
    print('--- Program initialization ---')
    init_program_parameters()
    print('--- Done')

    # Generation & loading data for learning & testing
    print('\n--- Data initialization ---')
    program_parameters[DATA], program_parameters[LABELS] = parse_data()
    print('--- Done')

    # Train model & visualize result
    print('\n--- Model training & saving ---')
    if arguments.create:
        model_parameters[MODEL], model_parameters[HISTORY] = \
            create_model(program_parameters[DATA], program_parameters[LABELS])
    print('--- Done')

    # Loading model
    print('\n--- Model loading ---')
    model_parameters[MODEL], model_parameters[HISTORY] = load_model_history()
    visualize(model_parameters[HISTORY])
    print('--- Done')

    # Test model
    print('\n--- Model testing ---')
    if arguments.test:
        print('--- Whole set')
        test_data(model_parameters[MODEL])
    print('--- Single case')
    cur_path = os.getcwd()
    predict_case(model_parameters[MODEL], arguments.image)
    print('--- Done')

