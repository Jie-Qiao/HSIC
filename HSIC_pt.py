from __future__ import division
import numpy as np
from scipy.stats import gamma
import torch


def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = torch.sum(pattern1 **2, 1).reshape(size1[0], 1)
    H = torch.sum(pattern2 **2, 1).reshape(size2[0], 1)

    Q = torch.repeat_interleave(G, repeats=size2[0],dim=1)

    R = torch.repeat_interleave(H.T, repeats=size1[0], dim=0)

    H = Q + R - 2 * torch.matmul(pattern1, pattern2.T)

    H = torch.exp(-H / 2 / (deg ** 2))

    return H


def hsic_gam(X, Y,width_x=None,width_y=None):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.shape[0]
    # ----- width of X -----
    if width_x is None:
        Xmed = X

        G = torch.sum(Xmed **2, 1).reshape(n, 1)
        Q = torch.repeat_interleave(G, n, dim=1)
        R = torch.repeat_interleave(G.T, n, dim=0)

        dists = Q + R - 2 * torch.matmul(Xmed, Xmed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n ** 2, 1)
        del G,Q,R
        torch.cuda.empty_cache()

        width_x = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
        del dists
        torch.cuda.empty_cache()
    # ----- -----

    # ----- width of Y -----
    if width_y is None:
        Ymed = Y

        G = torch.sum(Ymed **2, 1).reshape(n, 1)
        Q = torch.repeat_interleave(G, n,dim=1)
        R = torch.repeat_interleave(G.T, n,dim=0)

        dists = Q + R - 2 * torch.matmul(Ymed, Ymed.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(n ** 2, 1)
        del G,Q,R
        torch.cuda.empty_cache()
        width_y = torch.sqrt(0.5 * torch.median(dists[dists > 0]))
        del dists
        torch.cuda.empty_cache()
    # ----- -----

    bone = torch.ones(n, 1).to(X.device)
    H = torch.eye(n).to(X.device) - torch.ones(n, n).to(X.device) / n


    K = rbf_dot(X, X, width_x)
    torch.cuda.empty_cache()
    L = rbf_dot(Y, Y, width_y)
    torch.cuda.empty_cache()

    Kc = torch.matmul(torch.matmul(H, K), H).cpu()
    Lc = torch.matmul(torch.matmul(H, L), H).cpu()
    del H
    torch.cuda.empty_cache()

    testStat = torch.sum(Kc.T * Lc) / n
    testStat=testStat.item()

    varHSIC = (Kc * Lc / 6) ** 2
    del Kc,Lc
    torch.cuda.empty_cache()
    varHSIC = varHSIC.to(X.device)

    varHSIC = (torch.sum(varHSIC) - torch.trace(varHSIC)) / n / (n - 1)

    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K - torch.diag(torch.diag(K))
    L = L - torch.diag(torch.diag(L))

    muX = torch.matmul(torch.matmul(bone.T, K), bone) / n / (n - 1)
    muY = torch.matmul(torch.matmul(bone.T, L), bone) / n / (n - 1)
    del K,L
    torch.cuda.empty_cache()
    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC
    al=al.cpu().numpy()
    bet=bet.cpu().numpy()
    pval = 1 - gamma.cdf(testStat,al, scale=bet)[0][0]
    #alph=pval
    #thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]

    return testStat, pval


if __name__ == "__main__":
    # X = torch.randn(1000, 2)
    # Y = torch.randn(1000, 1)
    # print(hsic_gam(X,Y))

    X=torch.tensor([[0.1,0.1,1.0,2.0,3.0,4.0,5.0,2.0,3.0]]).T.float()
    Y=torch.tensor([[0.2,0.2,1.0,1.0,2.0,2.0,3.0,2.0,3.0]]).T.float()
    print(hsic_gam(X, Y))
