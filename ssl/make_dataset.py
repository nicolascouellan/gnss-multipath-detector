import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def load_data():
    # gps dataset
    # %% prepare sx3 dataset
    dataset_nomp = SX3Dataset(
        label=0, global_path=data_path + "sx_data/snap_no_mp_SX3_5_sat_11_89x81"
    )
    dataset_mp = SX3Dataset(
        label=1, global_path=data_path + "sx_data/snap_mp_SX3_5_sat_11_89x81"
    )
    data_nomp = dataset_nomp.build(discr_shape=(40, 40), nb_samples=NTOTAL // 2)  # 100
    data_mp = dataset_mp.build(discr_shape=(40, 40), nb_samples=NTOTAL // 2)  # 100

    data_val = np.concatenate((data_mp, data_nomp), axis=0)
    np.random.shuffle(data_val)

    X_val = np.array([x["table"] for x in data_val])
    X_mp = np.array([x["table"] for x in data_mp])
    X_nomp = np.array([x["table"] for x in data_nomp])
    y_val = np.array([x["label"] for x in data_val])[..., None]

    # create unsupervised targets: sup {-1,1}, unsup{0}
    unsup_ids = [
        random.randint(0, y_val.shape[0] - 1) for _ in range(y_val.shape[0] - NSUP)
    ]
    y_val[unsup_ids] = -1

    Xtrain, Xval, ytrain, yval = train_test_split(
        X_val, y_val, shuffle=True, test_size=0.15
    )

    # get ids of only yval sup
    Xval = Xval[(yval != -1).nonzero()[0]]
    yval = yval[(yval != -1).nonzero()[0]]

    print(Counter(ytrain.squeeze())), print(Counter(yval.squeeze()))
    print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)


def make_dataloaders():
    # introduce indices in dataset
    indices_train = np.arange(0, Xtrain.shape[0])
    indices_val = np.arange(0, Xval.shape[0])

    Xtrain_tensor = torch.tensor(Xtrain)
    ytrain_tensor = torch.tensor(ytrain)
    Xtrain_tensor = Xtrain_tensor.permute(0, 3, 1, 2)
    indices_tensor = torch.tensor(np.arange(0, Xtrain.shape[0]))
    train_tensor = TensorDataset(Xtrain_tensor, ytrain_tensor, indices_tensor)

    Xval_tensor = torch.tensor(Xval)
    yval_tensor = torch.tensor(yval)
    Xval_tensor = Xval_tensor.permute(0, 3, 1, 2)
    indices_val_tensor = torch.tensor(np.arange(0, Xval.shape[0]))
    val_tensor = TensorDataset(Xval_tensor, yval_tensor, indices_val_tensor)

    # train_sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=BATCH_SIZE, shuffle=False)
