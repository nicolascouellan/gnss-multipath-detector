import numpy as np
import ot


NUMITERMAX = 100
METHOD = "emd"


# compute wass distance for batch of 4 imgs
def funcWD(inpimg1, inpimg2, lambd=1e5):
    img1 = inpimg1[0, 15:25, 5:20]
    img2 = inpimg2[0, 15:25, 5:20]

    img1 = img1 + abs(img1.min())
    img1 /= img1.sum()
    img2 = img2 + abs(img2.min())
    img2 /= img2.sum()

    n = img1.flatten().shape[0]
    x = np.arange(n, dtype=np.float64)

    # loss matrix
    matr = ot.dist(x.reshape(n, 1), x.reshape(n, 1))
    matr /= matr.max()

    if METHOD == "sinkhorn":
        # solve sinkhorn. entropic regularized OT
        coupling = ot.sinkhorn(img1.flatten(), img2.flatten(), matr, lambd)
        wdist = ot.sinkhorn2(img1.flatten(), img2.flatten(), matr, lambd)
    elif METHOD == "emd":
        # solve EMD. exact linear program
        coupling = ot.emd(img1.flatten(), img2.flatten(), matr, numItermax=NUMITERMAX)
        wdist = ot.emd2(img1.flatten(), img2.flatten(), matr, numItermax=NUMITERMAX)
    else:
        raise ValueError("Bad method argument: ", METHOD)

    del img1, img2

    return wdist, coupling, matr


def batchWD(xbatch, lambd_wd=0.1):
    dists = np.zeros(shape=(xbatch.shape[0], xbatch.shape[0]))
    for i in range(xbatch.shape[0]):
        for j in range(xbatch.shape[0]):
            wdist, coupling, matr = funcWD(xbatch[i], xbatch[j], lambd=lambd_wd)
            dists[i, j] = wdist
    return dists
