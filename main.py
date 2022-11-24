import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Module & Dataset
import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import math
import random
import argparse
import sys
from models.DCEC import DCEC
from models.DEC import DEC
from models.N2D import N2D
from models.kmeans import kmeans
from models.metrics import Metrs
metrics = Metrs()

parser = argparse.ArgumentParser(prog='TSC', description='Two-Steps-Clustering')
parser.add_argument('-n', '--big-clusters', action='store', type=int, choices=range(10,100),
                   metavar="[10-100]", help='the number of big-clusters (final clusters), must be bigger than m', default=10)
parser.add_argument('--without-tsc', action='store_true', help='runs without tsc. ignores -m/--semi-clusters if this is true.')
parser.add_argument('-m', '--semi-clusters', action='store', type=int, choices=range(11,1000),
                   metavar="[11-1000]", required=True, help="the number of semi-clusters")
parser.add_argument('--latent-dim', action='store', type=int, choices=range(1,1000),
                   metavar="[1-1000]", default=10)
parser.add_argument('-p', '--dataset', action='store', type=str, choices=['mnist','fashion'], default='mnist')
parser.add_argument('-t', '--model-type', action='store', type=str, choices=['dcec','dec', 'kmeans', 'n2d'], default='dcec')
parser.add_argument('-V', '--verbose', action='store', type=int, choices=[0,1,2], default=0)
parser.add_argument('-l', '--load', action='store_true', help='load checkpoint data. you need to specify directory to load data. ignores -s/--save if this is true.')
parser.add_argument('-s', '--save', action='store_true', help='save checkpoint data. you need to specify directory to save data.')
parser.add_argument('-d', '--rw-dir', action='store', type=str, help='directory to load/save data')
parser.add_argument('--sa-param-a', action='store', type=int, nargs=1, default=80)
parser.add_argument('--sa-param-b', action='store', type=int, nargs=4, default=[-60, -90, -120, -150])
parser.add_argument('--sa-param-c', action='store', type=int, nargs=1, default=10000)
args = parser.parse_args()

assert args.big_clusters<args.semi_clusters

if args.dataset == 'mnist':
    (x_train, label_train), (x_test, label_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    X_TRAIN = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
elif args.dataset == 'fashion':
    (x_train, label_train), (x_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    X_TRAIN = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
else:
    raise Exception("dataset error")
if args.verbose>=2: print(x_train.shape, label_train.shape, x_test.shape, label_test.shape)

if args.model_type=='dcec':
    Y_PRED = DCEC(args,x_train,label_train)
if args.model_type=='dec':
    Y_PRED = DEC(args,x_train,label_train)
if args.model_type=='n2d':
    Y_PRED = N2D(args,x_train,label_train)
if args.model_type=='kmeans':
    Y_PRED = kmeans(args,x_train,label_train)

if args.without_tsc:
    sys.exit(0)

train = np.array(X_TRAIN)
label = np.array(Y_PRED)

S = []
models = []

for i in range(6):
    if args.verbose>=1: print("--------Model #%d--------" %i)
    newModel = tf.keras.models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(args.n_clusters, activation='softmax')
    ])
    newModel.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    newModel.fit(train[i*args.sa_param_c:(i+1)*args.sa_param_c], label[i*args.sa_param_c:(i+1)*args.sa_param_c], epochs=50, verbose=(args.verbose>=2), validation_data=(train, np.array(Y_PRED)))
    models.append(newModel)

cnt = [[[0 for _ in range(args.n_clusters)] for __ in range(args.n_clusters)] for ___ in range(4)]

for t in range(6):
    L = np.argmax(models[t].predict(train), axis=1)
    for i in range(60000):
        if L[i] != Y_PRED[i]:
            if t in [0, 1, 2]:
                cnt[0][L[i]][Y_PRED[i]]+=1
                cnt[0][Y_PRED[i]][L[i]]+=1
            if t in [0, 3, 4]:
                cnt[1][L[i]][Y_PRED[i]]+=1
                cnt[1][Y_PRED[i]][L[i]]+=1
            if t in [1, 3, 5]:
                cnt[2][L[i]][Y_PRED[i]]+=1
                cnt[2][Y_PRED[i]][L[i]]+=1
            if t in [2, 4, 5]:
                cnt[3][L[i]][Y_PRED[i]]+=1
                cnt[3][Y_PRED[i]][L[i]]+=1

SAparam = [args.sa_param_a, args.sa_param_b]
def scoreBigCluster(cluster, x, cntList):
    n = len(cluster)
    score = 0
    for i in range(n):
        for j in range(i+1, n):
            score += cntList[cluster[i]][cluster[j]]
    return score + x*(n**2)

def scoring(state, x, cntList):
    score = 0
    bigClusters = [[] for _ in range(10)]
    for i in range(args.n_clusters):
        bigClusters[state[i]].append(i)
    for s in bigClusters:
        score += scoreBigCluster(s, x, cntList)
    return score

def SA(t, d, k, lim, x, cntList):
    state = [random.randint(0, 9) for _ in range(args.n_clusters)]
    pos, next = 0, 0
    cnt = 0
    maxScore, maxState = -1000000000000000000000, []
    while t>lim:
        e1 = scoring(state, x, cntList)
        if maxScore < e1:
            maxScore = e1
            maxState = state[:]
            prLabels = np.array([state[Y_PRED[i]] for i in range(len(Y_PRED))])
            if (args.verbose>=2): print(f"* {cnt}: {e1} | %.5f %.5f %.5f"%(metrics.acc(label_train, prLabels), metrics.nmi(label_train, prLabels), metrics.ari(label_train, prLabels)))
        pos = (pos+1)%args.n_clusters
        if pos == 0:
            next = (next + 1) % 9
        if cnt%10000 == 0:
            prLabels = np.array([state[Y_PRED[i]] for i in range(len(Y_PRED))])
            if (args.verbose>=2):print(f"  {cnt}: {e1} | %.5f %.5f %.5f"%(metrics.acc(label_train, prLabels), metrics.nmi(label_train, prLabels), metrics.ari(label_train, prLabels)))
        prevState, newState = state[pos], (state[pos]+next+1)%10
        state[pos] = newState
        e2 = scoring(state, x, cntList)
        p = math.exp((e2-e1)/(k*t))
        if p<random.random():
            state[pos] = prevState
        t *= d
        cnt+=1
    return maxState

states = []
params = SAparam[1]
for i in range(4):
    if (args.verbose>=1): print(f"# Model {i+1}/4")
    states.append(SA(1, 0.999999, SAparam[0], 0.3, params[i], cnt[i]))
    if i>0:
        w = np.zeros((10, 10), dtype=np.int64)
        for j in range(args.n_clusters):
            w[states[i][j], states[0][j]] += 1
        ind = linear_assignment(w.max() - w)
        for j in range(args.n_clusters):
            states[i][j] = int(ind[1][np.where(ind[0]==states[i][j])])
    if (args.verbose>=2): print(states[i])

realState = []
for i in range(args.n_clusters):
    L = [0] * 10
    for j in range(4):
        L[states[j][i]] += 1
    realState.append(L.index(max(L)))

prLabels = np.array([realState[Y_PRED[i]] for i in range(len(Y_PRED))])

print("%.5f %.5f %.5f"%(metrics.acc(label_train, prLabels), metrics.nmi(label_train, prLabels), metrics.ari(label_train, prLabels)))
