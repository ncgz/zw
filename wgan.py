from __future__ import print_function, division
import argparse
import scipy
import tensorflow
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv3DTranspose, Subtract, Multiply, Lambda, GaussianNoise, SeparableConv1D, AveragePooling3D, Add, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling3D, Conv3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import datetime
import matplotlib.pyplot as plt
import scipy.io as scio
from glob import glob
import numpy as np
import os
import random
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
import tensorflow.keras.backend as K


class DataLoader():
    def __init__(self,  img_res=(80, 96, 80)):
        self.img_res = img_res

    def load_batch(self, batch_size):
        path = glob('D:/MRIHAR/trainforwgan2/*mat')
        random.shuffle(path)
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, fss, ses, mas = [], [], [], []

            for img in batch:
                img_A, fs, se, ma = self.imread(img)
                img_A = img_A.reshape((80, 96, 80, 1))
                #mask = img_A.reshape((96, 96, 96, 1))
                fs = fs.reshape((1))
                # fs=fs*(1-abs(0.1*np.random.randn(1)))
                se = se.reshape((1))
                # se=se*(1-abs(0.1*np.random.randn(1)))
                ma = ma.reshape((1))

                # ma=ma*(1-abs(0.1*np.random.randn(1)))
                imgs_A.append(img_A)
                fss.append(fs)
                ses.append(se)
                mas.append(ma)

            yield imgs_A, fss, ses, mas

    def load_val_batch(self):
        batch_size = 1
        path = glob('D:/MRIHAR/testforwgan2/*.mat')
        print(path)
        n_batches = 765
        for i in range(n_batches):
            batch = path[i]
            imgs_A, fss, ses, mas = [], [], [], []
            img_A, fs, se, ma = self.imread(batch)
            img_A = img_A.reshape((80, 96, 80, 1))
            imgs_A.append(img_A)
            fss.append(fs)
            ses.append(se)
            mas.append(ma)

            yield imgs_A, fss, ses, mas, batch

    def imread(self, path):
        k = scio.loadmat(path)
        img = k['img']
        fs = k['fs']
        se = k['se']
        ma = k['ma']

        return img, fs, se, ma
#


class HarmonyGan():

    def __init__(self):
        # Input shape
        self.img_shape = (80, 96, 80, 1)
        # Configure data loader
        self.data_loader = DataLoader(img_res=(80, 96, 80))

        # Calculate output shape of D (PatchGAN)
        # Number of filters in the first layer of G and D
        self.gf = 8
        self.df = 8
        optimizer_d = RMSprop(lr=0.00005)
        optimizer_g = RMSprop(lr=0.00005)
    #     optimizer_d = tensorflow.keras.optimizers.Adam(
    # learning_rate=0.0001,
    # beta_1=0.9,
    # beta_2=0.999,
    # epsilon=1e-07,
    # amsgrad=False,
    # name="Adam")
    #     optimizer_g = tensorflow.keras.optimizers.Adam(
    #         learning_rate=0.0001,
    #         beta_1=0.9,
    #         beta_2=0.999,
    #         epsilon=1e-07,
    #         amsgrad=False,
    #         name="Adam")
        # 学习率请一定一样
        # Build and compile the discriminator

        def wasserstein_loss(y_true, y_pred):
            return K.mean(y_true*y_pred)

        def cos(y_true, y_pred):
            y1 = K.reshape(y_true, (16, 614400))
            y2 = K.reshape(y_pred, (16, 614400))
            return -K.mean(K.batch_dot(y1, y2, axes=1) / (K.sqrt(K.batch_dot(y1, y1, axes=1)) * K.sqrt(K.batch_dot(y2, y2, axes=1))))

        def maee(y_true, y_pred):
            s = K.mean(K.abs(y_true-y_pred))
            s = K.exp(10*K.abs(s-0.1))
            return s

        def acc(y_true, y_pred):
            # Calculates the precision
            true_positives = 3+K.sum(K.sign(y_true * y_pred))/2
            #predicted_positives = K.sum(K.round(K.clip(y_pred, -1, 1)))
            accc = true_positives / 6
            return accc

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=[wasserstein_loss, wasserstein_loss, wasserstein_loss],
            # loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'],
            loss_weights=[1, 1, 1],
            optimizer=optimizer_d
        )

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_A)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        [v1, v2, v3] = self.discriminator([fake_A])

        self.combined = Model(inputs=[img_A], outputs=[
                              v1, v2, v3, fake_A, fake_A])
        self.combined.compile(loss=[wasserstein_loss, wasserstein_loss, wasserstein_loss, 'mae', cos],
                              # self.combined.compile(loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy', 'binary_crossentropy','binary_crossentropy', 'mae'],
                              loss_weights=[1, 1, 1, 0, 0],
                              # 100000太高
                              # 1000 97 太高
                              # 100 90
                              # 1 太低
                              # D十次，G一次效果还可以
                              # D需要dense之前
                              optimizer=optimizer_g
                              )

        # mae约为0.21
        # cos约为0.7~0.8

    def build_generator(self):
        """U-Net Generator"""

        def mul(x):
            t = scio.loadmat('D:\MRIHAR\mask.mat')
            imgg = t['img80']
            imgg = imgg.reshape((1, 80, 96, 80, 1))
            imgg = tensorflow.convert_to_tensor(imgg, dtype='float32')
            return tensorflow.multiply(x, imgg)

        def conv2d(layer_input, filters, f_size=3):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(layer_input)
            # d = InstanceNormalization()(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(d)
            # d = InstanceNormalization()(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)

            return d

        def conv2d2(layer_input, filters, f_size=3):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(layer_input)
            # d = InstanceNormalization()(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def conv2dd(layer_input, filters, f_size=3):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling3D(size=2)(layer_input)
            u = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(u)
            u = BatchNormalization()(u)
            # u = InstanceNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            u = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(u)
            u = BatchNormalization()(u)
            # u = InstanceNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            u = Concatenate()([u, skip_input])
            u = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(u)
            u = BatchNormalization()(u)
            # u = InstanceNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            u = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(u)
            u = BatchNormalization()(u)
            # u = InstanceNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        dd0 = GaussianNoise(0.1)(d0)
        # dd0= Conv3D(self.gf*8, kernel_size=1, strides=1, padding='same')(dd0)
        # dd0 = Conv3D(self.gf * 8, kernel_size=1, strides=1, padding='same')(dd0)
        # Downsampling
        d1 = conv2dd(dd0, self.gf)  # 40
        dd1 = MaxPooling3D(pool_size=(2, 2, 2),
                           strides=None, padding="valid")(d1)
        d2 = conv2d(dd1, self.gf * 2)  # 20
        dd2 = MaxPooling3D(pool_size=(2, 2, 2),
                           strides=None, padding="valid")(d2)
        d3 = conv2d(dd2, self.gf * 4)  # 10
        dd3 = MaxPooling3D(pool_size=(2, 2, 2),
                           strides=None, padding="valid")(d3)
        d4 = conv2d(dd3, self.gf * 8)  # 5
        dd4 = MaxPooling3D(pool_size=(2, 2, 2),
                           strides=None, padding="valid")(d4)

        dd4 = Conv3D(self.gf * 8, kernel_size=3,
                     strides=1, padding='same')(dd4)
        dd4 = BatchNormalization()(dd4)
        dd4 = LeakyReLU(alpha=0.2)(dd4)
        dd4 = Conv3D(self.gf * 8, kernel_size=3,
                     strides=1, padding='same')(dd4)
        dd4 = BatchNormalization()(dd4)
        dd4 = LeakyReLU(alpha=0.2)(dd4)
        # Upsampling
        u2 = deconv2d(dd4, d4, self.gf * 8)
        u3 = deconv2d(u2, d3, self.gf * 4)
        u4 = deconv2d(u3, d2, self.gf * 2)
        u5 = deconv2d(u4, d1, self.gf)
        u5 = Concatenate()([u5, d0])
        output_img = Conv3D(1, kernel_size=1, strides=1, padding='same')(u5)
        output_img = Lambda(mul, output_shape=(80, 96, 80, 1))(output_img)
        # #output_img2=Subtract()([d0,output_img])
        # output_img2=Lambda(mul, output_shape=(80,96,80, 1))(output_img)
        Model(d0, output_img).summary()
        # #Model(d0, [output_img,output_img2]).summary()
        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=3):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(layer_input)
            d = BatchNormalization()(d)
            # d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(d)
            d = BatchNormalization()(d)
            # d = InstanceNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = MaxPooling3D(pool_size=(2, 2, 2),
                             strides=None, padding="valid")(d)
            return d

        def d_layerr(layer_input, filters, f_size=3):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = Conv3D(filters, kernel_size=f_size,
                       strides=1, padding='same')(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = MaxPooling3D(pool_size=(2, 2, 2),
                             strides=None, padding="valid")(d)
            return d

        img_A = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        d0 = GaussianNoise(0.1)(img_A)
        d1 = d_layerr(d0, self.df * 1)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)
        q = Flatten()(d4)
        q = Dense(256)(q)
        # q=LeakyReLU(alpha=0.2)(q)
        # v1 = Dense(1,activation='sigmoid')(q)
        #q = Conv3D(1, kernel_size=1, strides=1, padding='same')(d4)
        # q=LeakyReLU(alpha=0.2)(q)
        #q = BatchNormalization()(q)
        # q = Flatten()(d4)
        # q = Dense(1024)(q)
        # q=LeakyReLU(alpha=0.2)(q)
        # q = BatchNormalization()(q)
        v1 = Dense(1)(q)
        # v1=ReLU(max_value=1)(v1)
        v2 = Dense(1)(q)
        # v2=ReLU(max_value=1)(v2)
        v3 = Dense(1)(q)
        #v4 = Dense(1)(q)
        # v3=ReLU(max_value=1)(v3)
        #v5 = ReLU(max_value=1)(v5)
        #v4 = Dense(3, activation='softmax')(q)
        Model(img_A, [v1, v2, v3]).summary()
        return Model(img_A, [v1, v2, v3])

    def train(self, epochs, batch_size, sample_interval=50):
        def wasserstein_loss(y_true, y_pred):
            return K.mean(y_true*y_pred)

        def cos(y_true, y_pred):
            y1 = K.reshape(y_true, (batch_size, 614400))
            y2 = K.reshape(y_pred, (batch_size, 614400))
            return -K.mean(K.batch_dot(y1, y2, axes=1) / (K.sqrt(K.batch_dot(y1, y1, axes=1)) * K.sqrt(K.batch_dot(y2, y2, axes=1))))
        start_time = datetime.datetime.now()
        optimizer_g = RMSprop(lr=0.00005)
        # Adversarial loss ground truths
        #valid =np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))],axis=1)
        #fake = np.concatenate([np.zeros((batch_size,1)),np.ones((batch_size,1))],axis=1)
        # fs2=np.concatenate([np.zeros((batch_size,1)),np.ones((batch_size,1))-np.abs(0.1*np.random.randn(batch_size,1))],axis=1)
        # se2=np.concatenate([np.ones((batch_size,1))-np.abs(0.1*np.random.randn(batch_size,1)),np.zeros((batch_size,1))],axis=1)
        #ma2 = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size,1))-np.abs(0.1*np.random.randn(batch_size,1)),np.zeros((batch_size, 1))], axis=1)
        #mloss2 = np.zeros((batch_size,80,96,80,1))

        #valid=-np.ones((batch_size, 1))
        #fake=np.ones((batch_size, 1))
        #mloss2 = np.zeros((batch_size, 80, 96, 80, 1))
        # sixteen=np.concatenate(np.ones(1),np.zeros(15))

        for epoch in range(epochs):
            m = 0
            q1 = []
            q2 = []
            q3 = []

            for batch_i, (imgs_A, fs, se, ma) in enumerate(self.data_loader.load_batch(batch_size)):

                imgs_A = np.array(imgs_A)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                fs = np.array(fs)
                fs = fs*(1-np.random.randn(batch_size)*0.1)
                se = np.array(se)
                se = se*(1-np.random.randn(batch_size)*0.1)
                ma = np.array(ma)
                ma = ma*(1-np.random.randn(batch_size)*0.1)
                # fs2 = -np.ones((batch_size, 1))
                # fs2=fs2*(1-np.random.randn(batch_size)*0.1)
                # se2 = -np.ones((batch_size, 1))
                # se2=se2*(1-np.random.randn(batch_size)*0.1)
                # ma2 = -np.ones((batch_size, 1))
                # ma2=ma2*(1-np.random.randn(batch_size)*0.1)

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_A)
                # Train the discriminators (original images = real / generated = Fake)

                d_loss = self.discriminator.train_on_batch(
                    imgs_A, [fs, se, ma])

                # d_loss_real = self.discriminator.train_on_batch(imgs_A, [valid,fs, se, ma])
                # d_loss_fake = self.discriminator.train_on_batch(fake_A, [fake,fs, se, ma])
                # #d_loss_real = self.discriminator.train_on_batch(imgs_A, [fs, se, ma])
                # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # pinimg = np.concatenate((imgs_A, fake_A), axis=0)
                # #pinva = np.concatenate((valid, fake), axis=0)
                # pinfs = np.concatenate((fs, fs), axis=0)
                # pinse = np.concatenate((se, se), axis=0)
                # pinma = np.concatenate((ma, ma), axis=0)

                # print(np.shape(pinimg))
                # print(np.shape(pinva))

                #d_loss = self.discriminator.train_on_batch(pinimg, [pinfs,pinse,pinma])
                # -----------------
                #  Train Generator
                # -----------------
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -0.1, 0.1) for w in weights]
                    l.set_weights(weights)
                # Train the generators
                q1.append(d_loss)

                g_loss = self.combined.train_on_batch(
                    imgs_A, [-fs, -se, -ma, imgs_A, imgs_A])
                m = m+np.abs(g_loss[1]+g_loss[2]+g_loss[3])
                # q=q+1
                # if q==5:
                #     q=0
                #     g_loss = self.combined.train_on_batch(imgs_A, [-fs,-se,-ma,imgs_A,imgs_A])
                # else:
                #     g_loss=np.zeros((6))

                #g_loss = self.combined.train_on_batch(imgs_A, [ mloss2])
                q2.append(g_loss)
                #q4.append(100 * d_loss[8])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print(
                    "[Epoch %d/%d] [Batch %d/%d]\n [D loss:%f,fs:%f,se:%f,ma:%f]\n[G loss1:%f,fs:%f,se:%f,ma:%f,mae:%f,cos:%f] time: %s" % (
                        epoch, epochs,
                        batch_i, self.data_loader.n_batches,
                        d_loss[0], d_loss[1], d_loss[2], d_loss[3],
                        #d_loss_fake[0], d_loss_fake[1], d_loss_fake[2], d_loss_fake[3],d_loss_fake[4],
                        g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5],
                        elapsed_time))
                # print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%,acc: %3d%%,acc: %3d%%] [G loss: %f,%3d%%,%3d%%,%3d%%,%f] time: %s" % (epoch, epochs,
                # batch_i, self.data_loader.n_batches,
                # d_loss[0], 100*d_loss[4],100*d_loss[5],100*d_loss[6],
                # g_loss[0], 100*g_loss[5], 100*g_loss[6], 100*g_loss[7], g_loss[3],
                # elapsed_time))
                # print(
                #     "[Epoch %d/%d] [Batch %d/%d]\n [D loss:%f, fs:%f,se:%f,ma:%f, acc: %3d%%,acc: %3d%%,acc: %3d%%]\n[D loss:%f, fs:%f,se:%f,ma:%f, acc: %3d%%,acc: %3d%%,acc: %3d%%]\n[G loss:%f, acc: %3d%%] time: %s" % (
                #         epoch, epochs,
                #         batch_i, self.data_loader.n_batches,
                #         d_loss_real[0], d_loss_real[1], d_loss_real[2], d_loss_real[3], 100 * d_loss_real[4],
                #         100 * d_loss_real[5], 100 * d_loss_real[6],
                #         d_loss_fake[0], d_loss_fake[1], d_loss_fake[2], d_loss_fake[3], 100 * d_loss_fake[4],
                #         100 * d_loss_fake[5], 100 * d_loss_fake[6],
                #         g_loss[0], g_loss[1], elapsed_time))
                # print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%,acc: %3d%%,acc: %3d%%] [G loss: %f,%3d%%,%3d%%,%3d%%,%f] time: %s" % (epoch, epochs,
                # batch_i, self.data_loader.n_batches,
                # d_loss[0], 100*d_loss[4],100*d_loss[5],100*d_loss[6],
                # g_loss[0], 100*g_loss[5], 100*g_loss[6], 100*g_loss[7], g_loss[3],
                # elapsed_time))
                # If at save interval => save generated image samples
                # if batch_i % sample_interval == 0:
                #     self.sample_images(epoch, batch_i)
            a1 = 'D:/MRIHAR/model2/dis_epoch' + str(epoch) + '.h5'
            self.discriminator.save(a1)
            a1 = 'D:/MRIHAR/model2/combined_epoch' + str(epoch) + '.h5'
            self.combined.save(a1)
            a2 = 'D:/MRIHAR/model2/a_epoch'+str(epoch)+'.mat'
            scipy.io.savemat(a2, {'q1': q1, 'q2': q2})

            self.combined.compile(loss=[wasserstein_loss, wasserstein_loss, wasserstein_loss, 'mae', cos],
                                  # self.combined.compile(loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy', 'binary_crossentropy','binary_crossentropy', 'mae'],
                                  loss_weights=[1, 1, 1, m/283, m/283],
                                  optimizer=optimizer_g
                                  )
            # for batch_i, (imgs_A, fs, se, ge, si, ph, batch) in enumerate(self.data_loader.load_val_batch()):
            #     imgs_A = np.array(imgs_A)
            #     real_loss = self.discriminator.predict_on_batch(imgs_A)
            #     print('fs:%f,%f\n'%(fs,real_loss[0]))
            #     print('se:%f,%f\n'%(se,real_loss[1]))
            #     print('ge:%f,%f\n'%(ge,real_loss[2]))
            #     print('si:%f,%f\n'%(si,real_loss[3]))
            #     print('ph:%f,%f\n'%(ph,real_loss[4]))

            #

            if np.mod(epoch+1, 5) == 0 or epoch <= 9:
                for batch_i, (imgs_A, fs, se, ma, batch) in enumerate(self.data_loader.load_val_batch()):
                    # fs22 = np.concatenate([np.zeros((1, 1)), np.ones((1, 1))], axis=1)
                    # se22 = np.concatenate([np.ones((1, 1)), np.zeros((1, 1))], axis=1)
                    # ma22 = np.concatenate([np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 1))], axis=1)
                    # mloss22 = np.zeros((1, 80, 96, 80, 1))
                    imgs_A = np.array(imgs_A)
                    d_loss = self.generator.predict_on_batch(imgs_A)
                    #real_loss = self.discriminator.predict_on_batch(imgs_A)
                    #fake_loss = self.combined.predict_on_batch(imgs_A)
                    print(batch)
                    a2 = 'D:/MRIHAR/2test-har' + \
                        str(epoch + 1) + '/' + \
                        batch[23:len(batch) - 4] + '.mat'
                    print(a2)
                    scipy.io.savemat(a2, {'img': d_loss[0]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=14, type=int)
    parser.add_argument("--sample_interval", default=200, type=int)
    config = parser.parse_args()

    gan = HarmonyGan()
    gan.train(epochs=config.epochs, batch_size=config.batch_size,
              sample_interval=config.sample_interval)
