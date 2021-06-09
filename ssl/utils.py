import numpy as np
import matplotlib.pyplot as plt

import plotly as py
import plotly.graph_objs as go
import pickle
import glob
import pandas as pd
from tqdm.notebook import tqdm
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from sklearn.metrics import accuracy_score, f1_score

# update path to import module from drive
import sys

sys.path.append("/content/drive/My Drive/04_these_ssl/")


def custom_subsample(arr, kernel_size=(3, 3)):
    # subsample each part separately
    kernel_x, kernel_y = kernel_size
    img_x, img_y = arr.shape[:2]
    result = np.zeros((arr.shape[0] // kernel_x, arr.shape[1] // kernel_y))
    i = 0
    j = 0
    ip = 0
    jp = 0
    while i + kernel_x <= arr.shape[0]:
        j = 0
        jp = 0
        while j + kernel_y <= arr.shape[1]:
            img_slice = arr[i : i + kernel_x, j : j + kernel_y]
            # print(img_slice)
            # result[ip,jp] = img_slice[img_slice.shape[0] // 2, img_slice.shape[1] // 2]
            result[ip, jp] = img_slice[0, 0]
            j += kernel_y
            jp += 1
            # print(i, j)
        i += kernel_x
        ip += 1

    return result


def custom_img_resize(arr, kernel_size):
    xmax, ymax = np.unravel_index(arr.argmax(), arr.shape)

    # split matrix by 4 slices
    upleft = arr[:xmax, :ymax]
    upright = arr[:xmax, ymax:]
    botleft = arr[xmax:, :ymax]
    botright = arr[xmax:, ymax:]

    # subsample each part separately
    sub_upleft = custom_subsample(upleft, kernel_size=kernel_size)
    sub_upright = custom_subsample(upright, kernel_size=kernel_size)
    sub_botleft = custom_subsample(botleft, kernel_size=kernel_size)
    sub_botright = custom_subsample(botright, kernel_size=kernel_size)

    # concat the images back
    uptemp = np.concatenate([sub_upleft, sub_upright], axis=1)
    bottemp = np.concatenate([sub_botleft, sub_botright], axis=1)
    res = np.concatenate([uptemp, bottemp], axis=0)
    return res


def load_ds_data(discr, data_path, nb_samples=None, test_size=0.05):
    """prepare data generator data """
    global_path_mp_i = data_path + "mp/*_i_*"  # .format(discr)
    global_path_mp_q = data_path + "mp/*_q_*"  # .format(discr)
    global_path_nomp_i = data_path + "no_mp/*_i_*"  # .format(discr)
    global_path_nomp_q = data_path + "no_mp/*_q_*"  # .format(discr)
    if nb_samples is None:
        paths_mp_i = sorted(glob.glob(global_path_mp_i))
        paths_mp_q = sorted(glob.glob(global_path_mp_q))
        paths_nomp_i = sorted(glob.glob(global_path_nomp_i))
        paths_nomp_q = sorted(glob.glob(global_path_nomp_q))
    else:
        paths_mp_i = sorted(glob.glob(global_path_mp_i))[: nb_samples // 2]
        paths_mp_q = sorted(glob.glob(global_path_mp_q))[: nb_samples // 2]
        paths_nomp_i = sorted(glob.glob(global_path_nomp_i))[: nb_samples // 2]
        paths_nomp_q = sorted(glob.glob(global_path_nomp_q))[: nb_samples // 2]

    synth_data_samples_mp = []
    synth_data_labels = []
    for path_mp_i, path_mp_q in tqdm(zip(paths_mp_i, paths_mp_q)):
        matr_i = pd.read_csv(path_mp_i, sep=",", header=None).values
        matr_q = pd.read_csv(path_mp_q, sep=",", header=None).values
        matr_i = cv2.resize(matr_i, (discr, discr))
        matr_q = cv2.resize(matr_q, (discr, discr))

        matr_i = matr_i[..., None]
        matr_q = matr_q[..., None]
        matr = np.concatenate((matr_i, matr_q), axis=2)
        # matr = matr_i**2 + matr_q**2
        synth_data_samples_mp.append(matr)
        synth_data_labels.append(1)

    synth_data_samples_nomp = []
    for path_nomp_i, path_nomp_q in tqdm(zip(paths_nomp_i, paths_nomp_q)):
        matr_i = pd.read_csv(path_nomp_i, sep=",", header=None).values
        matr_q = pd.read_csv(path_nomp_q, sep=",", header=None).values
        matr_i = cv2.resize(matr_i, (discr, discr))
        matr_q = cv2.resize(matr_q, (discr, discr))

        matr_i = matr_i[..., None]
        matr_q = matr_q[..., None]
        matr = np.concatenate((matr_i, matr_q), axis=2)
        # matr = matr_i**2 + matr_q**2
        synth_data_samples_nomp.append(matr)
        synth_data_labels.append(0)

    synth_data_samples = np.concatenate(
        [synth_data_samples_mp, synth_data_samples_nomp], axis=0
    )
    synth_data_labels = np.array(synth_data_labels)

    X_train_synth, X_val_synth, y_train_synth, y_val_synth = train_test_split(
        synth_data_samples, synth_data_labels, test_size=test_size, shuffle=True
    )

    return X_train_synth, X_val_synth, y_train_synth, y_val_synth


def build_model(backbone, input_shape, is_trainable=True):
    """
    implementation taken from pudae
    https://github.com/pudae/kaggle-hpa/blob/master/models/model_factory.py
    """
    for layer in backbone.layers:
        layer.trainable = is_trainable
    model = Sequential()
    model.add(
        Conv2D(3, (3, 3), activation="relu", padding="same", input_shape=(80, 80, 2))
    )
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))
    return model


def model_eval(model, X, y, model_name, threshold):
    probas = model.predict(X)
    preds = np.where(probas >= threshold, 1, 0)
    acc = accuracy_score(preds, y)
    f1 = f1_score(preds, y)
    print("best acc: {:.5}, best f1: {:.5}".format(acc, f1))
    model.save("best_{}_{:.5f}.h5".format(model_name, acc))


def save_model(model, file_name):
    """
    Save model in pickle format
    :param model: training model
    :param file_name: String
        Path to the model file
    """
    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def load_model(file_name):
    """
    Load model from file
    :param file_name: String
        Path to the model file
    :return: predictive model
    """
    with open(file_name, "rb") as file:
        return pickle.load(file)


def visualize_plt(data_sample):
    size0, size1 = data_sample.shape[0], data_sample.shape[1]

    # Plot 2D
    plt.imshow(data_sample)
    plt.show()

    # Plot 3D
    # fig = plt.figure()
    # x = np.linspace(0, size0-1, size0)
    # y = np.linspace(0, size1-1, size1)
    # print(len(x), len(y))
    # print(data_sample.shape)
    # ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    # cset = ax.contour3D(x, y, data_sample, 800)
    # ax.clabel(cset, fontsize=9, inline=1)
    # plt.show()


def visualize_3d_discr(
    func,
    discr_size_fd,
    scale_code,
    tau_interv,
    dopp_interv,
    Tint,
    delta_dopp=0,
    delta_tau=0,
    alpha_att=1,
    delta_phase=0,
    filename="3d_surface_check_discr.html",
):

    y = np.linspace(tau_interv[0], tau_interv[1], discr_size_fd)
    x = np.linspace(dopp_interv[0], dopp_interv[1], scale_code)

    data = [go.Surface(x=x, y=y, z=func)]

    layout = go.Layout(
        title="3d surface check_discr",
        autosize=True,
        xaxis=go.layout.XAxis(range=[-1000, 1000]),
        yaxis=go.layout.YAxis(range=[-1000, 1000]),
        scene=dict(
            yaxis=dict(
                nticks=10, range=[y.min(), y.max()], title="Pixels_X (Code Delay [s])"
            ),
            xaxis=dict(
                nticks=10, range=[x.min(), x.max()], title="Pixels_Y (Doppler [Hz])"
            ),
            zaxis=dict(nticks=10, range=[func.min(), func.max()]),
            annotations=[
                dict(
                    showarrow=False,
                    x=x.max(),
                    y=y.max(),
                    z=func.max(),
                    text="Tint = {}s".format(Tint),
                    xanchor="left",
                    xshift=10,
                ),
                dict(
                    showarrow=False,
                    x=x.max(),
                    y=y.max(),
                    z=func.max() * 0.95,
                    text="delta_dopp = {} Hz".format(delta_dopp),
                    xanchor="left",
                    xshift=10,
                ),
                dict(
                    showarrow=False,
                    x=x.max(),
                    y=y.max(),
                    z=func.max() * 0.9,
                    text="delta_tau = {} s".format(delta_tau),
                    xanchor="left",
                    xshift=10,
                ),
                dict(
                    showarrow=False,
                    x=x.max(),
                    y=y.max(),
                    z=func.max() * 0.85,
                    text="alpha_att = {}".format(alpha_att),
                    xanchor="left",
                    xshift=10,
                ),
                dict(
                    showarrow=False,
                    x=x.max(),
                    y=y.max(),
                    z=func.max() * 0.8,
                    text="delta_phase = {} deg".format(delta_phase * 180 / np.pi),
                    xanchor="left",
                    xshift=10,
                ),
            ],
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename=filename)
