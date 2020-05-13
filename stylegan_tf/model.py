import numpy as np
import tensorflow as tf


def D_stylegan2(
        images_in,  # First input: Images [minibatch, channel, height, width].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1), act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 2), kernel=3, down=True, resample_kernel=resample_kernel),
                               act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res - 2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out
