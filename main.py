import cv2
import os
import numpy as np

# suppress info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HEIGHT   = 1072
WIDTH    = 1920
CHANNELS = 3

scenes = ['Gura_Portitei_1', 'Gura_Portitei_2', 'Delta_Crisan_1']

X_train = {}
y_train = {}

for scene in scenes:
    scene_path = f'dataset/{scene}_train'

    path = f'{scene_path}/images'

    filenames = [filename for filename in os.listdir(path)]

    X_train[scene] = np.zeros((len(filenames), HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
    y_train[scene] = np.zeros((len(filenames), HEIGHT, WIDTH, 1), dtype=np.bool)

    for i, filename in enumerate(filenames):
        image_path = f'{scene_path}/images/{filename}'
        mask_path  = f'{scene_path}/masks/{filename}'

        X_train[scene][i] = cv2.imread(image_path)
        y_train[scene][i] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).reshape((HEIGHT, WIDTH, 1))

# get model
import tensorflow as tf
from model import UNet

models = {scene: UNet(HEIGHT, WIDTH, CHANNELS) for scene in scenes}

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 3)

for scene in scenes:
    models[scene].fit(X_train[scene], y_train[scene], batch_size=1, epochs=100, callbacks=[callback])

# train a model using all the data
models['all'] = UNet(HEIGHT, WIDTH, CHANNELS)

X_train_all = None
y_train_all = None

for scene in scenes:
    if X_train_all is None and y_train_all is None:
        X_train_all = X_train[scene]
        y_train_all = y_train[scene]
    else:
        X_train_all = np.vstack((X_train_all, X_train[scene]))
        y_train_all = np.vstack((y_train_all, y_train[scene]))

models['all'].fit(X_train_all, y_train_all, batch_size=1, epochs=100, callbacks=[callback])

# compute videos
for scene in scenes:
    for data_type in ['train', 'test']:
        path = f'{scene}_{data_type}'

        video_path  = f'videos/{path}.mp4'

        video = cv2.VideoCapture(video_path)

        success = True
        count = 0

        writer_scene = cv2.VideoWriter(f'results/{scene}_{data_type}_scene.mp4', -1, 30, (WIDTH, HEIGHT))
        writer_all   = cv2.VideoWriter(f'results/{scene}_{data_type}_all.mp4', -1, 30, (WIDTH, HEIGHT))

        while success:
            count += 1

            print(scene, data_type, count)

            success, image = video.read()

            if image is None:
                break

            image = np.reshape(image[4:1076, :, :], (1, HEIGHT, WIDTH, CHANNELS))
            pred_scene = np.reshape(models[scene].predict(image), (WIDTH, HEIGHT))
            pred_all = np.reshape(models['all'].predict(image), (WIDTH, HEIGHT))

            writer_scene.write(np.array([[[255, 255, 255] if pred_scene[i][j] else [0, 0, 0] for j in range(HEIGHT)] for i in range(WIDTH)]).astype(np.uint8))
            writer_all.write(np.array([[[255, 255, 255] if pred_all[i][j] else [0, 0, 0] for j in range(HEIGHT)] for i in range(WIDTH)]).astype(np.uint8))

        writer_scene.release()
        writer_all.release()