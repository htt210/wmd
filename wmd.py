#start
import sys, numpy as np, pickle
from emd import emd

load_file = sys.argv[1]
save_file = sys.argv[2]

with open(load_file) as f:
    [X, BOW_X, y, C, words] = pickle.load(f)

n = np.shape(X)[0]
for i in range(n):
    BOW_X[i] /= np.sum(BOW_X[i])
    X_i = X[i].T
    X_i = X_i.tolist()
    X[i] = X_i


def get_wmd(i):

    Di = np.zeros((1,n))
    for j in range(n):
        if len(X[i]) > 0 and len(X[j]) > 0:
            #print i, j, len(X[i]), len(X[j])
            Di[0,j] = emd(X[i], X[j], X_weights=BOW_X[i], Y_weights=BOW_X[j])
        else:
            Di[0,j] = 2.0
    return Di


def main():
    with open(save_file, 'w') as f:
        pickle.dump(get_wmd(0), f)


if __name__ == "__main__":
    main()
