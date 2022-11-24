# Import Module & Dataset
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Activation, LeakyReLU, Layer
from keras import backend as K
from keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from .metrics import Metrs
metrics = Metrs()

def DEC(args, x_train, label_train) -> list:
    """
    DEC
    """

    args.n_clusters = args.semi_clusters
    args.n_final_clusters = args.big_clusters
    args.batch_size       = 256        # type=int
    args.epochs           = 3e2        # type=int **
    args.maxiter          = 2e4        # type=int **
    args.gamma            = 0          # type=float, help='coefficient of clustering loss'
    args.update_interval  = 150        # type=int, help='This argument must be given'
    args.tol              = 1e-5       # type=float
    args.n_init           = 20         # type=int, help='클러스터링 초기값 정할때 몇번 돌릴지 결정'


    args.epochs = int(args.epochs)
    args.maxiter = int(args.maxiter)

    """
    Define Model
    """

    class DCEC(Model):

        class Autoencoder:
            """
            Autoencoder represents a Deep Convolutional autoencoder architecture with
            mirrored encoder and decoder components.
            """

            def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
                self.input_shape = input_shape
                self.conv_filters = conv_filters
                self.conv_kernels = conv_kernels
                self.conv_strides = conv_strides
                self.latent_space_dim = latent_space_dim

                self.encoder = None
                self.decoder = None
                self.model = None

                self._num_conv_layers = len(conv_filters)
                self._shape_before_bottleneck = None
                self._model_input = None

                self._build()

            def summary(self):
                self.encoder.summary()
                self.decoder.summary()
                self.model.summary()
            
            def compile(self, learning_rate=0.0001):
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                mse_loss = MeanSquaredError()
                self.model.compile(optimizer=optimizer, loss=mse_loss)

            def train(self, x_train, batch_size, num_epochs):
                self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=(args.verbose>=2))

            def _build(self):
                self._build_encoder()
                self._build_decoder()
                self._build_autoencoder()

            def _build_autoencoder(self):
                model_input = self._model_input
                model_output = self.decoder(self.encoder(model_input))
                self.model = Model(model_input, model_output, name="autoencoder")

            def _build_decoder(self):
                decoder_input = self._add_decoder_input()
                dense_layer = self._add_dense_layer(decoder_input)
                reshape_layer = self._add_reshape_layer(dense_layer)
                conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
                decoder_output = self._add_decoder_output(conv_transpose_layers)
                self.decoder = Model(decoder_input, decoder_output, name="decoder")

            def _add_decoder_input(self):
                return Input(shape=self.latent_space_dim, name="decoder_input")

            def _add_dense_layer(self, decoder_input):
                num_neurons = np.prod(self._shape_before_bottleneck)
                dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
                return dense_layer

            def _add_reshape_layer(self, dense_layer):
                return Reshape(self._shape_before_bottleneck)(dense_layer)

            def _add_conv_transpose_layers(self, x):
                """Add conv transpose blocks."""
                # loop through all the conv layers in reverse order and stop at the
                # first layer
                for layer_index in reversed(range(1, self._num_conv_layers)):
                    x = self._add_conv_transpose_layer(layer_index, x)
                return x

            def _add_conv_transpose_layer(self, layer_index, x):
                layer_num = self._num_conv_layers - layer_index
                conv_transpose_layer = Conv2DTranspose(
                    filters=self.conv_filters[layer_index],
                    kernel_size=self.conv_kernels[layer_index],
                    strides=self.conv_strides[layer_index],
                    padding="same",
                    name=f"decoder_conv_transpose_layer_{layer_num}"
                )
                x = conv_transpose_layer(x)
                x = LeakyReLU(0.01, name=f"decoder_relu_{layer_num}")(x)
                x = tf.keras.layers.BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
                return x

            def _add_decoder_output(self, x):
                conv_transpose_layer = Conv2DTranspose(
                    filters=1,
                    kernel_size=self.conv_kernels[0],
                    strides=self.conv_strides[0],
                    padding="same",
                    name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
                )
                x = conv_transpose_layer(x)
                output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
                return output_layer

            def _build_encoder(self):
                encoder_input = self._add_encoder_input()
                conv_layers = self._add_conv_layers(encoder_input)
                bottleneck = self._add_bottleneck(conv_layers)
                self._model_input = encoder_input
                self.encoder = Model(encoder_input, bottleneck, name="encoder")

            def _add_encoder_input(self):
                return Input(shape=self.input_shape, name="encoder_input")

            def _add_conv_layers(self, encoder_input):
                """Create all convolutional blocks in encoder."""
                x = encoder_input
                for layer_index in range(self._num_conv_layers):
                    x = self._add_conv_layer(layer_index, x)
                return x

            def _add_conv_layer(self, layer_index, x):
                """Add a convolutional block to a graph of layers, consisting of
                conv 2d + ReLU + batch normalization.
                """
                layer_number = layer_index + 1
                conv_layer = Conv2D(
                    filters=self.conv_filters[layer_index],
                    kernel_size=self.conv_kernels[layer_index],
                    strides=self.conv_strides[layer_index],
                    padding="same",
                    name=f"encoder_conv_layer_{layer_number}"
                )
                x = conv_layer(x)
                x = LeakyReLU(0.01, name=f"encoder_relu_{layer_number}")(x)
                x = tf.keras.layers.BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
                return x

            def _add_bottleneck(self, x):
                """Flatten data and add bottleneck (Dense layer)."""
                self._shape_before_bottleneck = K.int_shape(x)[1:]
                x = Flatten()(x)
                x = Dense(self.latent_space_dim, name="encoder_output")(x)
                return x

        class ClusterLayer(Layer):

            class ClusterLayerInitializer(tf.keras.initializers.Initializer):
                def __init__(self, encoder, n):
                    self.encoder = encoder
                    self.n_clusters = n

                def __call__(self, shape, dtype=None, **kwargs):
                    self.z = self.encoder(x_train)
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=args.n_init).fit(self.z)
                    returnVal = tf.convert_to_tensor(kmeans.cluster_centers_, dtype=tf.float32)
                    del kmeans
                    return returnVal

            def __init__(self, n_clusters, init_encoder, ni):
                super(DCEC.ClusterLayer, self).__init__()
                self.n_clusters = n_clusters
                self.init_encoder = init_encoder
                self.ni = ni
            
            def build(self, input_shape):
                if args.verbose>=1 and self.ni: print("...Building a cluster layer...")
                self.N, self.feature_dim = input_shape
                self.mu = self.add_weight(shape=(self.n_clusters, self.feature_dim),
                                        initializer=DCEC.ClusterLayer.ClusterLayerInitializer(self.init_encoder, self.n_clusters) if self.ni else "zeros",
                                        trainable=True,
                                        name='mu'
                                        )
                self.built = True
            
            def call(self, feature_data):
                q = 1.0 / (1.0 + K.sum(K.square(K.expand_dims(feature_data, axis=1) - self.mu), axis=2))
                q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
                return q

        def __init__(self, n_clusters, latent_dim):
            super(DCEC, self).__init__()
            self.n_clusters = n_clusters
            self.latent_dim = latent_dim

        def make_model(self, x_train, label_train, optimizer='adam', batch_size=args.batch_size, epochs=100):
            self.N_train = x_train.shape[0]
            if args.verbose>=1: print("...Making a model...")
            inputs = tf.keras.Input(shape=x_train.shape[1:]+[1])
            if args.load:
                self.autoencoder = DCEC.Autoencoder(input_shape=tuple(list(x_train.shape[1:])+[1]), conv_filters=(32, 64, 128), conv_kernels=(5, 5, 5), conv_strides=(2, 2, 1), latent_space_dim=self.latent_dim)
                print('loading model from:', args.rw_dir+f'/AE_model.h5')
                self.autoencoder.model.load_weights(args.rw_dir+f'/AE_model.h5')
            else:
                self.autoencoder = DCEC.Autoencoder(input_shape=x_train.shape[1:]+[1], conv_filters=(32, 64, 128), conv_kernels=(5, 5, 5), conv_strides=(2, 2, 1), latent_space_dim=self.latent_dim)
                self.autoencoder.compile(0.001)
                self.autoencoder.train(x_train, batch_size, epochs)
            self.encoder = self.autoencoder.encoder(inputs)
            self.decoder = self.autoencoder.decoder(self.encoder)
            self.clusteringLayer = DCEC.ClusterLayer(self.n_clusters, self.autoencoder.encoder, ni=True)
            self.clusterLayer = self.clusteringLayer(self.encoder)
            self.model = Model(inputs=inputs, outputs=[self.clusterLayer, self.decoder])

        def set_model(self, input_shape):
            inputs = tf.keras.Input(shape=input_shape)
            self.autoencoder = DCEC.Autoencoder(input_shape=tuple(list(x_train.shape[1:])+[1]), conv_filters=(32, 64, 128), conv_kernels=(5, 5, 5), conv_strides=(2, 2, 1), latent_space_dim=self.latent_dim)
            self.encoder = self.autoencoder.encoder(inputs)
            self.decoder = self.autoencoder.decoder(self.encoder)
            self.clusteringLayer = DCEC.ClusterLayer(self.n_clusters, self.autoencoder.encoder, ni=False)
            self.clusterLayer = self.clusteringLayer(self.encoder)
            self.model = Model(inputs=inputs, outputs=[self.clusterLayer, self.decoder])

        def compile(self, optimizer, loss, gamma):
            self.model.compile(optimizer=optimizer, loss=loss, loss_weights = [gamma, 1])
        
        @staticmethod
        def sliceTensor(x, ite, batch_size, N):
            index = (ite * batch_size) % N
            if index + batch_size <= N:
                return x[index : index+batch_size]
            return tf.concat([x[index : ], x[ : index+batch_size-N]], axis=0)

        def fit(self, x_train, label_train, maxiter=args.maxiter, batch_size=args.batch_size, tol=args.tol, T=args.update_interval):
            loss = 0
            pred_train_last = np.zeros((self.N_train,))
            pred_train = self.model.predict(x_train, verbose=(args.verbose>=2))[0].argmax(1)
            for ite in range(int(maxiter)):
                if ite % T == 0:
                    q = self.model.predict(x_train, verbose=(args.verbose>=2))[0]
                    pred_train = q.argmax(1)
                    weight = q**2 / tf.reduce_sum(q, axis = 0)
                    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis = 1))

                    # evaluate the clustering performance
                    nmi = normalized_mutual_info_score(label_train, pred_train)
                    acc = 0
                    if args.verbose>=2:
                        print('Iter %d: acc=%.5f, nmi = %.5f, loss = %.5f' % (ite, acc, nmi, loss))

                    # check stop criterion
                    ne = tf.cast(tf.math.not_equal(pred_train,pred_train_last), tf.int32)
                    delta_label = tf.cast(tf.reduce_sum(ne), tf.float32) / pred_train.shape[0]
                    pred_train_last = tf.identity(pred_train)

                    if args.verbose>=1 and ite > 0 and delta_label < tol:
                        print('delta_label ', delta_label, '< tol', tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                loss = self.model.train_on_batch(x=DCEC.sliceTensor(x_train, ite, batch_size, self.N_train),
                                                y=[DCEC.sliceTensor(p, ite, batch_size, self.N_train), DCEC.sliceTensor(x_train, ite, batch_size, self.N_train)])[0]
            return pred_train

    """
    Make Model
    """

    dcec = DCEC(args.n_clusters, args.latent_dim)

    """
    Build Model
    """

    if args.load:
        print('loading model from:', args.rw_dir+f'/DCEC_model.h5')
        dcec.set_model(input_shape=tuple(list(x_train.shape[1:])+[1]))
        dcec.model.load_weights(args.rw_dir+f'/DCEC_model.h5')
    else:
        Tx_train = tf.convert_to_tensor(x_train)
        Tlabel_train = tf.convert_to_tensor(label_train)
        dcec.make_model(Tx_train, Tlabel_train, batch_size=args.batch_size, epochs=args.epochs)
        if args.save:
            print('saving model to:', args.rw_dir+f'/AE_model.h5')
            dcec.autoencoder.model.save_weights(args.rw_dir+f'/AE_model.h5')
        dcec.compile(optimizer='adam', loss=['kld', 'mse'], gamma=args.gamma)
        y_pred = dcec.fit(Tx_train, Tlabel_train)
        if args.save:
            print('saving model to:', args.rw_dir+f'/DCEC_model.h5')
            dcec.model.save_weights(args.rw_dir+f'/DCEC_model.h5')

    if args.verbose>=2: plot_model(dcec.model, show_shapes=True)

    """
    Save Y_PRED
    """

    y_pred = dcec.model.predict(x_train)[0]
    y_pred = [list(y_pred[i]).index(max(y_pred[i])) for i in range(len(y_pred))]
    if args.without_tsc: print("%.5f %.5f %.5f"%(metrics.acc(label_train, np.array(y_pred)), metrics.nmi(label_train, np.array(y_pred)), metrics.ari(label_train, np.array(y_pred))))

    return y_pred
