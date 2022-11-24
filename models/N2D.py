# Import Module & Dataset
import os
import random as rn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
import pandas as pd
import numpy as np
import sys
import umap.umap_ as umap
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from scipy.optimize import linear_sum_assignment as linear_assignment
from time import time
from .metrics import Metrs
metrics = Metrs()


def N2D(args, x_train, label_train) -> list:
    """
    N2D
    """

    args.batch_size       = 256         # type=int
    args.pretrain_epochs  = 1e3         # type=int **
    args.save             = False       # type=bool
    args.save_dir         = None        # type=str|None
    args.ae_weights       = None        # type=str|None
    args.cluster          = 'GMM'       # type=str
    args.eval_all         = False       # type=bool
    args.manifold_learner = 'UMAP'      # type=str
    args.umap_dim         = 2           # type=int
    args.umap_neighbors   = 10          # type=int
    args.umap_min_dist    = '0.00'      # type=str
    args.umap_metric      = 'euclidean' # type=str
    args.visualize        = False       # type=bool

    args.pretrain_epochs = int(args.pretrain_epochs)

    rn.seed(0)
    np.random.seed(0)

    #try:
    #    from MulticoreTSNE import MulticoreTSNE as TSNE
    #except BaseException:
    #    print("Missing MulticoreTSNE package.. Only important if evaluating other manifold learners.")
    TSNE = lambda x: x

    matplotlib.use('agg')

    def eval_other_methods(x, y, names=None):
        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=args.n_clusters,
            random_state=0)
        gmm.fit(x)
        y_pred_prob = gmm.predict_proba(x)
        y_pred = y_pred_prob.argmax(1)
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(args.dataset + " | GMM clustering on raw data")
        print('=' * 80)
        print(acc)
        print(nmi)
        print(ari)
        print('=' * 80)

        y_pred = KMeans(
            n_clusters=args.n_clusters,
            random_state=0).fit_predict(x)
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(args.dataset + " | K-Means clustering on raw data")
        print('=' * 80)
        print(acc)
        print(nmi)
        print(ari)
        print('=' * 80)

        sc = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(x)
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(args.dataset + " | Spectral Clustering on raw data")
        print('=' * 80)
        print(acc)
        print(nmi)
        print(ari)
        print('=' * 80)

        if args.manifold_learner == 'UMAP':
            md = float(args.umap_min_dist)
            hle = umap.UMAP(
                random_state=0,
                metric=args.umap_metric,
                n_components=args.umap_dim,
                n_neighbors=args.umap_neighbors,
                min_dist=md).fit_transform(x)
        elif args.manifold_learner == 'LLE':
            from sklearn.manifold import LocallyLinearEmbedding
            hle = LocallyLinearEmbedding(
                n_components=args.umap_dim,
                n_neighbors=args.umap_neighbors).fit_transform(x)
        elif args.manifold_learner == 'tSNE':
            method = 'exact'
            hle = TSNE(
                n_components=args.umap_dim,
                n_jobs=16,
                random_state=0,
                verbose=0).fit_transform(x)
        elif args.manifold_learner == 'isomap':
            hle = Isomap(
                n_components=args.umap_dim,
                n_neighbors=5,
            ).fit_transform(x)

        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=args.n_clusters,
            random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(args.dataset + " | GMM clustering on " +
            str(args.manifold_learner) + " embedding")
        print('=' * 80)
        print(acc)
        print(nmi)
        print(ari)
        print('=' * 80)

        if args.visualize:
            plot(hle, y, 'UMAP', names)
            y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
            plot(hle, y_pred_viz, 'UMAP-predicted', names)

            return

        y_pred = KMeans(
            n_clusters=args.n_clusters,
            random_state=0).fit_predict(hle)
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(args.dataset + " | K-Means " +
            str(args.manifold_learner) + " embedding")
        print('=' * 80)
        print(acc)
        print(nmi)
        print(ari)
        print('=' * 80)

        sc = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(args.dataset + " | Spectral Clustering on " +
            str(args.manifold_learner) + " embedding")
        print('=' * 80)
        print(acc)
        print(nmi)
        print(ari)
        print('=' * 80)

    def cluster_manifold_in_embedding(hl, y, label_names=None):
        # find manifold on autoencoded embedding
        if args.manifold_learner == 'UMAP':
            md = float(args.umap_min_dist)
            hle = umap.UMAP(
                random_state=0,
                metric=args.umap_metric,
                n_components=args.umap_dim,
                n_neighbors=args.umap_neighbors,
                min_dist=md).fit_transform(hl)
        elif args.manifold_learner == 'LLE':
            hle = LocallyLinearEmbedding(
                n_components=args.umap_dim,
                n_neighbors=args.umap_neighbors).fit_transform(hl)
        elif args.manifold_learner == 'tSNE':
            hle = TSNE(
                n_components=args.umap_dim,
                n_jobs=16,
                random_state=0,
                verbose=0).fit_transform(hl)
        elif args.manifold_learner == 'isomap':
            hle = Isomap(
                n_components=args.umap_dim,
                n_neighbors=5,
            ).fit_transform(hl)

        # clustering on new manifold of autoencoded embedding
        if args.cluster == 'GMM':
            gmm = mixture.GaussianMixture(
                covariance_type='full',
                n_components=args.n_clusters,
                random_state=0)
            gmm.fit(hle)
            y_pred_prob = gmm.predict_proba(hle)
            y_pred = y_pred_prob.argmax(1)
        elif args.cluster == 'KM':
            km = KMeans(
                init='k-means++',
                n_clusters=args.n_clusters,
                random_state=0,
                n_init=20)
            y_pred = km.fit_predict(hle)
        elif args.cluster == 'SC':
            sc = SpectralClustering(
                n_clusters=args.n_clusters,
                random_state=0,
                affinity='nearest_neighbors')
            y_pred = sc.fit_predict(hle)

        y_pred = np.asarray(y_pred)
        # y_pred = y_pred.reshape(len(y_pred), )
        y = np.asarray(y)
        # y = y.reshape(len(y), )
        
        # Skip acc calc

        if args.visualize:
            plot(hle, y, 'n2d', label_names)
            y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
            plot(hle, y_pred_viz, 'n2d-predicted', label_names)
        return y_pred, 0,0,0

    def best_cluster_fit(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = linear_assignment(w.max() - w)
        best_fit = []
        for i in range(y_pred.size):
            for j in range(len(ind)):
                if ind[j][0] == y_pred[i]:
                    best_fit.append(ind[j][1])
        return best_fit, ind, w

    def cluster_acc(y_true, y_pred):
        _, ind, w = best_cluster_fit(y_true, y_pred)
        ind=[(ind[0][i],ind[1][i],) for i in range(len(ind[0]))]
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def plot(x, y, plot_id, names=None):
        viz_df = pd.DataFrame(data=x[:5000])
        viz_df['Label'] = y[:5000]
        if names is not None:
            viz_df['Label'] = viz_df['Label'].map(names)
        viz_df.to_csv(args.save_dir + '/' + args.dataset + '.csv')
        plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=0, y=1, hue='Label', legend='full', hue_order=sorted(viz_df['Label'].unique()),
                        palette=sns.color_palette("hls", n_colors=args.n_clusters),
                        alpha=.5,
                        data=viz_df)
        l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                    mode="expand", borderaxespad=0, ncol=args.n_clusters + 1, handletextpad=0.01, )

        l.texts[0].set_text("")
        plt.ylabel("")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(args.save_dir + '/' + args.dataset +
                    '-' + plot_id + '.png', dpi=300)
        plt.clf()

    def autoencoder(dims, act='relu'):
        n_stacks = len(dims) - 1
        x = Input(shape=(dims[0],), name='input')
        h = x
        for i in range(n_stacks - 1):
            h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
        h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
        for i in range(n_stacks - 1, 0, -1):
            h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
        h = Dense(dims[0], name='decoder_0')(h)
        return Model(inputs=x, outputs=h)

    optimizer = 'adam'

    x,y = x_train, label_train
    if args.dataset=='fashion':
        label_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    else:
        label_names = None

    shape = [x.shape[-1], 500, 500, 2000, args.n_clusters]
    autoencoder = autoencoder(shape)

    hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden)

    pretrain_time = time()

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        autoencoder.compile(loss='mse', optimizer=optimizer)
        autoencoder.fit(
            x,
            x,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            verbose = (args.verbose>=2))
        pretrain_time = time() - pretrain_time
        if args.save:
            autoencoder.save_weights('weights/' +
                                        args.dataset +
                                        "-" +
                                        str(args.pretrain_epochs) +
                                        '-ae_weights.h5')
        print("Time to train the autoencoder: " + str(pretrain_time))
    else:
        autoencoder.load_weights('weights/' + args.ae_weights)

    if args.save:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        with open(args.save_dir + '/args.txt', 'w') as f:
            f.write("\n".join(sys.argv))

    hl = encoder.predict(x)
    if args.eval_all:
        eval_other_methods(x, y, label_names)
    clusters, t_acc, t_nmi, t_ari = cluster_manifold_in_embedding(
        hl, y, label_names)
    if args.save: np.savetxt(args.save_dir + "/" + args.dataset + '-clusters.txt', clusters, fmt='%i', delimiter=',')
    x_train,label_train,y_pred = x,y,clusters
    if args.without_tsc: print("%.5f %.5f %.5f"%(metrics.acc(label_train, np.array(y_pred)), metrics.nmi(label_train, np.array(y_pred)), metrics.ari(label_train, np.array(y_pred))))

    return y_pred[:60000]