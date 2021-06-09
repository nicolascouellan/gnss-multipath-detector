import torch
import pandas as pd
import torch.nn as nn


def train_val(model, lambd_loss=0, ssl=False):
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    for epoch in range(epochs):

        running_train_loss = 0
        running_train_corrects = 0
        # train phase
        model.train()
        for i, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            if ssl:
                # recompute distsi for batch
                dists = torch.tensor(batchWD(batch[0])).to(device)
                # dists = torch.rand((batch[0].shape[0], batch[0].shape[0]))
                output = model(batch[0].float().to(device))
                preds = torch.round(torch.sigmoid(output))
                loss = sslLoss(
                    output, batch[1].to(device), dists, lambd=lambd_loss, sigma=SIGMA
                )
            else:
                output = model(batch[0].float().to(device))
                # preds = (output > THR).long()
                preds = torch.round(torch.sigmoid(output))
                loss = criterion(output, batch[1].float().to(device))

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()  # * batch[0].size(0)

            # compute acc only for labeled images in each batch. get ids of only yval sup
            sup_batch = batch[1].data.cpu()
            sup_preds = preds.data.cpu()
            # print('check shape batch: ', sup_batch.shape, sup_preds.shape)
            sup_preds = sup_preds[(sup_batch != -1).numpy().nonzero()[0]]
            sup_batch = sup_batch[(sup_batch != -1).numpy().nonzero()[0]]
            # print('check shape sup batch: ', sup_batch.shape, sup_preds.shape)
            running_train_corrects += (
                torch.sum(sup_preds == sup_batch) / batch[1].shape[0]
            )

        nb_batches = i + 1
        losses.append(running_train_loss / nb_batches)
        accs.append(running_train_corrects / nb_batches)

        running_val_loss = 0
        running_val_corrects = 0
        # val phase
        model.eval()
        for i, batch in enumerate(loaders["val"]):
            output = model(batch[0].float().to(device))
            # preds = (output > THR).long()
            preds = torch.round(torch.sigmoid(output))

            loss = criterion(output, batch[1].float().to(device))

            running_val_loss += loss.item()  # * batch[0].size(0)

            # compute acc only for labeled images in each batch. get ids of only yval sup
            sup_batch = batch[1].data.cpu()
            sup_preds = preds.data.cpu()
            sup_preds = sup_preds[(sup_batch != -1).numpy().nonzero()[0]]
            sup_batch = sup_batch[(sup_batch != -1).numpy().nonzero()[0]]
            running_val_corrects += (
                torch.sum(sup_preds == sup_batch) / batch[1].shape[0]
            )

        nb_val_batches = i + 1
        val_losses.append(running_val_loss / nb_val_batches)
        val_accs.append(running_val_corrects / nb_val_batches)

        print("-----------------Epoch: {}".format(epoch))
        print(preds.squeeze()[:15], batch[1].squeeze()[:15])
        print("Train Loss: {}".format(round(running_train_loss / nb_batches, 5)))
        print(
            "Train Acc: {}".format(round(float(running_train_corrects) / nb_batches, 5))
        )
        print("Val Loss: {}".format(round(running_val_loss / nb_val_batches, 5)))
        print(
            "Val Acc: {}".format(round(float(running_val_corrects) / nb_val_batches, 5))
        )

    return model, losses, val_losses, accs, val_accs


# lambdas = [0.01, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.5, 0.8, 1, 2, 5, 10]
# lambd_loss = 5 #NSUP / (NTOTAL - NSUP)

val_loss_df = pd.DataFrame(columns=["valloss", "iter"])
val_acc_df = pd.DataFrame(columns=["valacc", "iter"])

for i in tqdm(range(NB_RUNS)):
    model = Model().to(device)

    loaders = {"train": train_loader, "val": val_loader}

    lr = 1e-3
    epochs = 16
    THR = 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("------------------------- lambda: {}".format(lambd_loss))
    model, losses, val_losses, accs, val_accs = train_val(
        model, lambd_loss=lambd_loss, ssl=True
    )
    val_accs = torch.tensor(val_accs)
    val_losses = torch.tensor(val_losses)

    val_loss_df.loc[i, ["valloss", "iter"]] = [val_losses.min(), val_losses.argmin()]
    val_acc_df.loc[i, ["valacc", "iter"]] = [val_accs.max(), val_accs.argmax()]
plot_conv_curves_bench(losses, val_losses, accs, val_accs)
