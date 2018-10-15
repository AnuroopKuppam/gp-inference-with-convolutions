import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
from matplotlib import gridspec
import itertools
import sys
from missing_observations import generate_data, plot, conv_kernel, squared_exponential_kernel, generate_noise


def convolution(N, kernel, noise, y, missing_indices):
    tf.reset_default_graph()
    gr = tf.get_default_graph()
    y_flat = tf.convert_to_tensor(np.ndarray.flatten(y))
    missing_indices = ~missing_indices
    mask = tf.convert_to_tensor(1.0 * np.ndarray.flatten(missing_indices))
    input_vector = np.array(kernel)

    # tensorflow
    # rehape k2
    k1, k2 = input_vector.shape
    input_vector = np.expand_dims(input_vector, axis=0)
    input_vector = np.expand_dims(input_vector, axis=3)

    # tensorflow
    y_prime = tf.layers.conv2d(
        inputs=tf.convert_to_tensor(input_vector),
        filters=1,
        kernel_size=[N, N],
        name="conv1",
        use_bias=False,
        padding="valid")

    # add the D.v term
    v_prime = gr.get_tensor_by_name('conv1/kernel:0')
    v = tf.reshape(v_prime, [N * N])
    y_prime += tf.reduce_sum(tf.multiply(noise, v), 1)
    # l2 loss
    loss = tf.nn.l2_loss((y_prime - y_flat) * mask)

    # conjugate gradient loss
    #   v_prime = gr.get_tensor_by_name('conv1/kernel:0')
    #   v = tf.reshape(v_prime, [N*N])
    #   y_prime = tf.reshape(y_prime, [N*N])
    #   first = 0.5 * tf.reduce_sum(tf.multiply(v, y_prime), 0)
    #   first += 0.5 * tf.reduce_sum(tf.multiply(v, tf.reduce_sum(tf.multiply(noise, v), 1)), 0)

    #   second = -1.0 * tf.reduce_sum(tf.multiply(v, y_flat), 0)
    #   loss = first + second

    # optimization step
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        loss_plot = []
        for i in range(10000):
            sess.run(optimizer)
            append = loss.eval({}, sess)
            # print(sess.run(conv1))
            # recover graph
            if i % 1000 == 0:
                gr = tf.get_default_graph()
                v = gr.get_tensor_by_name('conv1/kernel:0').eval()
                v = np.reshape(v, (N * N))
                print('loss: ', append)
                loss_plot.append(append)
                # plot new image
                img = y_prime.eval({}, sess)
                img = np.reshape(img, (N, N))
                plot(img)
                # img = np.dot(K, v)
                # plot(np.reshape(img, (N, N)))
    plt.plot(loss_plot)
    sess.close()
    return img


def main():
    y, miss = generate_data(style='parabola')
    K_prime = conv_kernel(y.shape[0])
    D = generate_noise(y.shape[0], miss)
    # K = squared_exponential_kernel(y.shape[0])
    y_prime = convolution(y.shape[0], K_prime, D, y, miss)
    plot(y_prime)
    return


if __name__ == '__main__':
    main()
