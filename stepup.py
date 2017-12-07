import numpy as np
import tensorflow as tf
ds = tf.contrib.distributions
import os
import glob
import imageio
import scipy.misc
import pickle
import math
from pathos.multiprocessing import ProcessPool
import h5py

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [200]")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate [0.0005]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta2", 0.999, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "batch size used in training [64]")
flags.DEFINE_integer("param_size",15, "batch size used in training [64]")
pool = ProcessPool()


FLAGS = flags.FLAGS

def load_real_images(): 
    f = h5py.File("../RenderGAN-tensorflow/data/beestags/real_tags.hdf5", "r")

    raw = f["rois"]
    return raw

def load():
    """load data and labels in parallel"""
    load_helper_im = lambda i: np.reshape(np.array(scipy.misc.imresize(scipy.misc.imread('./data/'+str(i)+'.png', mode='L').astype(np.float), [96, 96])), (96,96,1))/127.5 - 1.
    load_helper_p = lambda i: np.array(map(float, pickle.load(open("data/"+str(i)+".data", "rb"))))
    
    N = len(glob.glob('./data/*'))
    data = pool.map(load_helper_im, np.arange(150))
    labels = pool.map(load_helper_p,np.arange(150))
    return data,labels

def generate_sample_params():
    """generate parameters in the same style as the 3D model to produce test samples"""
    res = []
    for i in range(FLAGS.batch_size):       
        roll, pitch, yaw = np.random.uniform(360),np.random.uniform(45),np.random.uniform(45)
        tag = ''
        for i in range(12):
            tag += str(np.random.randint(2))
        params = map(float,list(tag))
        params.extend([roll, pitch, yaw])
        params=np.array(params)
        res.append(params)
    return res

def save_samples(imgs,count, prefix=''):
    """save generated samples"""
    imgs = (imgs+1)/2
    for i in range(FLAGS.batch_size):
        scipy.misc.imsave('samples/' + prefix + str(count) + '-' + str(i) + '.jpg',  np.reshape(imgs[i], (96,96)))
    

def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak*x)

def get_kernel_matrix(sigma):
    with tf.name_scope('kernel_matrix_' + str(sigma)):
        k = np.array([
            [6.,5.,4.,3.,4.,5.,6.],
            [5.,4.,3.,2.,3.,4.,5.],
            [4.,3.,2.,1.,2.,3.,4.],
            [3.,2.,1.,0.,1.,2.,3.],
            [4.,3.,2.,1.,2.,3.,4.],
            [5.,4.,3.,2.,3.,4.,5.],
            [6.,5.,4.,3.,4.,5.,6.]
            ]).reshape(7, 7, 1, 1)
        init_kernel_matrix = tf.Variable(k, dtype=tf.float32, trainable=False)
        intermediate_kernel_matrix = tf.exp(-init_kernel_matrix/(2*(sigma)**2))
        kernel_matrix = intermediate_kernel_matrix/tf.reduce_sum(intermediate_kernel_matrix)
        return kernel_matrix


def main(_):
    if not os.path.exists('logs/'):
        os.makedirs('logs/')
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')
    if not os.path.exists('samples/'):
        os.makedirs('samples/')

    print('loading data...')
    data,labels = load()
    real_images = load_real_images()
    print('loaded!')
    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    run_config.gpu_options.allow_growth=True

    """start graph setup"""
    with tf.variable_scope('model') as scope:
        params = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.param_size], name='params')
        FC1 = tf.contrib.layers.fully_connected(params, 1024, activation_fn=lrelu)
        FC2 = tf.contrib.layers.fully_connected(FC1, 6*6*256, activation_fn=lrelu)
        M = tf.reshape(FC2,[FLAGS.batch_size,6,6,256])
        DECONV1=tf.layers.conv2d_transpose(M,256,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='deconv1')
        DECONV2=tf.layers.conv2d_transpose(DECONV1,92,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='deconv2')
        DECONV3=tf.layers.conv2d_transpose(DECONV2,92,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='deconv3')
        generated_pics=tf.layers.conv2d_transpose(DECONV3,1,(5,5),strides=(2,2),padding='same', name='deconv4')

    actual_pics = tf.placeholder(tf.float32, [FLAGS.batch_size, 96,96,1], name='imgs')

    kernel_matrix = get_kernel_matrix(2.0)

    with tf.variable_scope('generator') as scope:
        with tf.variable_scope('blurring_augmentation') as scope:

            blur_alpha = tf.Variable(.5, name="blur_alpha", trainable = True)
            blurred = tf.nn.conv2d(generated_pics, kernel_matrix, strides = [1, 1, 1, 1], padding='SAME', name='blur')
            blurred_output = (1-blur_alpha)*(generated_pics-blurred) + blurred

        with tf.variable_scope('lighting_augmentation') as scope:

            s_w = tf.Variable(.7, name="s_b", trainable = True)
            s_b = tf.Variable(.7, name="s_b", trainable = True)
            s_t = tf.Variable(0.0, name="s_b", trainable = True)
            condition = tf.greater(blurred_output,0.0)
            W = tf.where(condition,tf.ones(condition.shape, dtype=tf.float32),tf.zeros(condition.shape, dtype=tf.float32))
            blurred_sw = tf.nn.conv2d(blurred_output*s_w, kernel_matrix, strides = [1, 1, 1, 1], padding='SAME', name='blur')
            blurred_sb = tf.nn.conv2d(blurred_output*s_b, kernel_matrix, strides = [1, 1, 1, 1], padding='SAME', name='blur')
            blurred_st = tf.nn.conv2d(blurred_output+s_t, kernel_matrix, strides = [1, 1, 1, 1], padding='SAME', name='blur')

            lighting_output = blurred_output*blurred_sw*W + blurred_output*blurred_sb*(1-W) + blurred_st

        kernel_matrix = get_kernel_matrix(3.5)
        with tf.variable_scope('highpass_augmentation') as scope:
            highpass_diff = .5*lighting_output - .5*tf.nn.conv2d(lighting_output, kernel_matrix, strides = [1, 1, 1, 1], padding='SAME', name='blur')
            generator_output = tf.clip_by_value(highpass_diff, -2, 2)


    with tf.variable_scope('discriminator') as scope:
        CONV1 = tf.layers.conv2d(actual_pics,92,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv1')
        CONV2 = tf.layers.conv2d(CONV1,92,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv2')
        CONV3 = tf.layers.conv2d(CONV2,256,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv3')
        CONV4 = tf.layers.conv2d(CONV3,256,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv4')
        L = tf.reshape(CONV4,[FLAGS.batch_size,6*6*256])
        h4 = tf.layers.dense(L,1,activation=lrelu,name='fc')
        d_real_out = h4


    with tf.variable_scope('discriminator') as scope:
        scope.reuse_variables()
        CONV1 = tf.layers.conv2d(generator_output,92,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv1')
        CONV2 = tf.layers.conv2d(CONV1,92,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv2')
        CONV3 = tf.layers.conv2d(CONV2,256,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv3')
        CONV4 = tf.layers.conv2d(CONV3,256,(5,5),strides=(2,2),padding='same', activity_regularizer=lrelu, name='conv4')
        L = tf.reshape(CONV4,[FLAGS.batch_size,6*6*256])
        h4 = tf.layers.dense(L,1,activation=lrelu,name='fc')
        d_fake_out = h4


    sq_loss = tf.reduce_mean(tf.squared_difference(actual_pics,generated_pics))

    loss_sum = tf.summary.scalar("squared_loss", sq_loss)


    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=d_real_out, logits=tf.ones_like(d_real_out)))
    d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=d_fake_out, logits=tf.zeros_like(d_fake_out)))
    g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=d_fake_out, logits=tf.ones_like(d_fake_out)))
    d_loss = d_loss_real + d_loss_fake

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)         
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)

    t_vars = tf.trainable_variables()
    m_vars = [var for var in t_vars if 'model/' in var.name]
    d_vars = [var for var in t_vars if 'discriminator/' in var.name]
    g_vars = [var for var in t_vars if 'generator/' in var.name]


    """start optimizer setup"""
    learning_rate_tensor = tf.Variable(FLAGS.learning_rate, trainable=False, name='lr', dtype=tf.float32)
    lr_sum = tf.summary.scalar("learning_rate", learning_rate_tensor)

    m_optim = tf.train.AdamOptimizer(learning_rate_tensor, beta1=FLAGS.beta1, beta2=FLAGS.beta2) \
              .minimize(sq_loss, var_list=m_vars)
    d_optim = tf.train.AdamOptimizer(learning_rate_tensor, beta1=FLAGS.beta1, beta2=FLAGS.beta2) \
              .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate_tensor, beta1=FLAGS.beta1, beta2=FLAGS.beta2) \
              .minimize(g_loss, var_list=g_vars)

    summaries = tf.summary.merge_all()


    with tf.Session(config=run_config) as sess:
        sess.run(tf.global_variables_initializer())
        
        """logger and saver"""
        summary_writer = tf.summary.FileWriter('logs/',sess.graph)
        saver = tf.train.Saver(max_to_keep=2*FLAGS.epoch, keep_checkpoint_every_n_hours=2)

        """start training"""
        iters = len(data)/FLAGS.batch_size

        for count in range(FLAGS.epoch*iters):
            index = count%iters
            ep = count/iters
            batch_params = labels[index*FLAGS.batch_size: index*FLAGS.batch_size+FLAGS.batch_size]
            batch_pics = data[index*FLAGS.batch_size: index*FLAGS.batch_size+FLAGS.batch_size]
            real_pics = np.array(real_images[index*FLAGS.batch_size: index*FLAGS.batch_size+FLAGS.batch_size]).reshape(FLAGS.batch_size, 96,96,1)
            sess.run(m_optim, feed_dict={ params: batch_params , actual_pics: batch_pics })
            sess.run(d_optim, feed_dict={ params: batch_params , actual_pics: real_pics })
            sess.run(g_optim, feed_dict={ params: batch_params , actual_pics: real_pics })

            if index==0:
                print('saving, epoch: ' + str(ep))
                print(str(learning_rate_tensor.eval()))
                saver.save(sess, 'checkpoint/model-' + str(count) + ".ckpt")
                sample_params = generate_sample_params()
                sample_pics = sess.run(generated_pics, feed_dict={ params: sample_params })
                save_samples(sample_pics,count,prefix='stepup-')
                sample_pics = sess.run(generator_output, feed_dict={ params: sample_params })
                save_samples(sample_pics,count,prefix='output-')

                summ = sess.run(summaries,feed_dict={ params: batch_params , actual_pics: real_pics })
                summary_writer.add_summary(summ, count)

                if (ep%50==0) and ep > 0:
                    learning_rate_tensor = learning_rate_tensor/2



if __name__ == '__main__':
  tf.app.run()
