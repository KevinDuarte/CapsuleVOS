import tensorflow as tf
import numpy as np
from functools import reduce
import config

epsilon = 1e-7


# Standard EM Routing
def em_routing(v, a_i, beta_v, beta_a, n_iterations=3):
    batch_size = tf.shape(v)[0]
    _, _, n_caps_j, mat_len = v.get_shape().as_list()
    n_caps_j, mat_len = map(int, [n_caps_j, mat_len])
    n_caps_i = tf.shape(v)[1]

    a_i = tf.expand_dims(a_i, axis=-1)

    # Prior probabilities for routing
    r = tf.ones(shape=(batch_size, n_caps_i, n_caps_j, 1), dtype=tf.float32)/float(n_caps_j)
    r = tf.multiply(r, a_i)

    den = tf.reduce_sum(r, axis=1, keep_dims=True) + epsilon

    # Mean: shape=(N, 1, Ch_j, mat_len)
    m_num = tf.reduce_sum(v*r, axis=1, keep_dims=True)
    m = m_num/(den + epsilon)

    # Stddev: shape=(N, 1, Ch_j, mat_len)
    s_num = tf.reduce_sum(r * tf.square(v - m), axis=1, keep_dims=True)
    s = s_num/(den + epsilon)

    # cost_h: shape=(N, 1, Ch_j, mat_len)
    cost = (beta_v + tf.log(tf.sqrt(s + epsilon) + epsilon)) * den
    # cost_h: shape=(N, 1, Ch_j, 1)
    cost = tf.reduce_sum(cost, axis=-1, keep_dims=True)

    # calculates the mean and std_deviation of the cost, used for numerical stability
    cost_mean = tf.reduce_mean(cost, axis=-2, keep_dims=True)
    cost_stdv = tf.sqrt(
        tf.reduce_sum(
            tf.square(cost - cost_mean), axis=-2, keep_dims=True
        ) / n_caps_j + epsilon
    )

    # calculates the activations for the capsules in layer j
    a_j = tf.sigmoid(float(config.inv_temp) * (beta_a + (cost_mean - cost) / (cost_stdv + epsilon)))

    def condition(mean, stdsqr, act_j, r_temp, counter):
        return tf.less(counter, n_iterations)

    def route(mean, stdsqr, act_j, r_temp, counter):
        exp = tf.reduce_sum(tf.square(v - mean) / (2 * stdsqr + epsilon), axis=-1)
        coef = 0 - .5 * tf.reduce_sum(tf.log(2 * np.pi * stdsqr + epsilon), axis=-1)
        log_p_j = coef - exp

        log_ap = tf.reshape(tf.log(act_j + epsilon), (batch_size, 1, n_caps_j)) + log_p_j
        r_ij = tf.nn.softmax(log_ap + epsilon)  # ap / (tf.reduce_sum(ap, axis=-1, keep_dims=True) + epsilon)

        r_ij = tf.multiply(tf.expand_dims(r_ij, axis=-1), a_i)

        denom = tf.reduce_sum(r_ij, axis=1, keep_dims=True) + epsilon
        m_numer = tf.reduce_sum(v * r_ij, axis=1, keep_dims=True)
        mean = m_numer / (denom + epsilon)

        s_numer = tf.reduce_sum(r_ij * tf.square(v - mean), axis=1, keep_dims=True)
        stdsqr = s_numer / (denom + epsilon)

        cost_h = (beta_v + tf.log(tf.sqrt(stdsqr) + epsilon)) * denom

        cost_h = tf.reduce_sum(cost_h, axis=-1, keep_dims=True)
        cost_h_mean = tf.reduce_mean(cost_h, axis=-2, keep_dims=True)
        cost_h_stdv = tf.sqrt(
            tf.reduce_sum(
                tf.square(cost_h - cost_h_mean), axis=-2, keep_dims=True
            ) / n_caps_j
        )

        inv_temp = config.inv_temp + counter * config.inv_temp_delta
        act_j = tf.sigmoid(inv_temp * (beta_a + (cost_h_mean - cost_h) / (cost_h_stdv + epsilon)))

        return mean, stdsqr, act_j, r_ij, tf.add(counter, 1)

    [mean, _, act_j, r_new, _] = tf.while_loop(condition, route, [m, s, a_j, r, 1.0])

    return tf.reshape(mean, (batch_size, n_caps_j, mat_len)), tf.reshape(act_j, (batch_size, n_caps_j, 1))


# Attention Routing layer
def em_routing_cond(v_v1, v_v2, a_i_v, v_f, a_i_f, beta_v_v, beta_a_v, beta_v_f, beta_a_f, n_iterations=3):
    batch_size = tf.shape(v_f)[0]
    _, _, n_caps_j, mat_len = v_f.get_shape().as_list()
    n_caps_j, mat_len = map(int, [n_caps_j, mat_len])
    n_caps_i_f = tf.shape(v_f)[1]

    a_i_f = tf.expand_dims(a_i_f, axis=-1)

    # Prior probabilities for routing
    r_f = tf.ones(shape=(batch_size, n_caps_i_f, n_caps_j, 1), dtype=tf.float32)/float(n_caps_j)
    r_f = tf.multiply(r_f, a_i_f)

    den_f = tf.reduce_sum(r_f, axis=1, keep_dims=True) + epsilon

    m_num_f = tf.reduce_sum(v_f*r_f, axis=1, keep_dims=True)  # Mean: shape=(N, 1, Ch_j, mat_len)
    m_f = m_num_f/(den_f + epsilon)

    s_num_f = tf.reduce_sum(r_f * tf.square(v_f - m_f), axis=1, keep_dims=True)  # Stddev: shape=(N, 1, Ch_j, mat_len)
    s_f = s_num_f/(den_f + epsilon)

    cost_f = (beta_v_f + tf.log(tf.sqrt(s_f + epsilon) + epsilon)) * den_f  # cost_h: shape=(N, 1, Ch_j, mat_len)
    cost_f = tf.reduce_sum(cost_f, axis=-1, keep_dims=True)  # cost_h: shape=(N, 1, Ch_j, 1)

    # calculates the mean and std_deviation of the cost
    cost_mean_f = tf.reduce_mean(cost_f, axis=-2, keep_dims=True)
    cost_stdv_f = tf.sqrt(
        tf.reduce_sum(
            tf.square(cost_f - cost_mean_f), axis=-2, keep_dims=True
        ) / n_caps_j + epsilon
    )

    # calculates the activations for the capsules in layer j for the frame capsules
    a_j_f = tf.sigmoid(float(config.inv_temp) * (beta_a_f + (cost_mean_f - cost_f) / (cost_stdv_f + epsilon)))

    def condition(mean_f, stdsqr_f, act_j_f, counter):
        return tf.less(counter, n_iterations)

    def route(mean_f, stdsqr_f, act_j_f, counter):
        # Performs E-step for frames
        exp_f = tf.reduce_sum(tf.square(v_f - mean_f) / (2 * stdsqr_f + epsilon), axis=-1)
        coef_f = 0 - .5 * tf.reduce_sum(tf.log(2 * np.pi * stdsqr_f + epsilon), axis=-1)
        log_p_j_f = coef_f - exp_f

        log_ap_f = tf.reshape(tf.log(act_j_f + epsilon), (batch_size, 1, n_caps_j)) + log_p_j_f
        r_ij_f = tf.nn.softmax(log_ap_f + epsilon)

        # Performs M-step for frames
        r_ij_f = tf.multiply(tf.expand_dims(r_ij_f, axis=-1), a_i_f)

        denom_f = tf.reduce_sum(r_ij_f, axis=1, keep_dims=True) + epsilon
        m_numer_f = tf.reduce_sum(v_f * r_ij_f, axis=1, keep_dims=True)
        mean_f = m_numer_f / (denom_f + epsilon)

        s_numer_f = tf.reduce_sum(r_ij_f * tf.square(v_f - mean_f), axis=1, keep_dims=True)
        stdsqr_f = s_numer_f / (denom_f + epsilon)

        cost_h_f = (beta_v_f + tf.log(tf.sqrt(stdsqr_f + epsilon) + epsilon)) * denom_f

        cost_h_f = tf.reduce_sum(cost_h_f, axis=-1, keep_dims=True)
        cost_h_mean_f = tf.reduce_mean(cost_h_f, axis=-2, keep_dims=True)
        cost_h_stdv_f = tf.sqrt(
            tf.reduce_sum(
                tf.square(cost_h_f - cost_h_mean_f), axis=-2, keep_dims=True
            ) / n_caps_j + epsilon
        )

        inv_temp = config.inv_temp + counter * config.inv_temp_delta
        act_j_f = tf.sigmoid(inv_temp * (beta_a_f + (cost_h_mean_f - cost_h_f) / (cost_h_stdv_f + epsilon)))

        return mean_f, stdsqr_f, act_j_f, tf.add(counter, 1)

    [mean_f_fin, _, act_j_f_fin, _] = tf.while_loop(condition, route, [m_f, s_f, a_j_f, 1.0])

    # performs m step for the video capsules
    a_i_v = tf.expand_dims(a_i_v, axis=-1)

    dist_v = tf.reduce_sum(tf.square(v_v1 - mean_f_fin), axis=-1)
    r_v = tf.expand_dims(tf.nn.softmax(0 - dist_v), axis=-1) * a_i_v

    den_v = tf.reduce_sum(r_v, axis=1, keep_dims=True) + epsilon

    m_num_v = tf.reduce_sum(v_v2 * r_v, axis=1, keep_dims=True)  # Mean: shape=(N, 1, Ch_j, mat_len)
    m_v = m_num_v / (den_v + epsilon)

    s_num_v = tf.reduce_sum(r_v * tf.square(v_v2 - m_v), axis=1, keep_dims=True)  # Stddev: shape=(N, 1, Ch_j, mat_len)
    s_v = s_num_v / (den_v + epsilon)

    cost_v = (beta_v_v + tf.log(tf.sqrt(s_v + epsilon) + epsilon)) * den_v  # cost_h: shape=(N, 1, Ch_j, mat_len)
    cost_v = tf.reduce_sum(cost_v, axis=-1, keep_dims=True)  # cost_h: shape=(N, 1, Ch_j, 1)

    # calculates the mean and std_deviation of the cost
    cost_mean_v = tf.reduce_mean(cost_v, axis=-2, keep_dims=True)
    cost_stdv_v = tf.sqrt(
        tf.reduce_sum(
            tf.square(cost_v - cost_mean_v), axis=-2, keep_dims=True
        ) / n_caps_j + epsilon
    )

    # calculates the activations for the capsules in layer j for the frame capsules
    a_j_v = tf.sigmoid(float(config.inv_temp) * (beta_a_v + (cost_mean_v - cost_v) / (cost_stdv_v + epsilon)))

    return (tf.reshape(m_v, (batch_size, n_caps_j, mat_len)), tf.reshape(a_j_v, (batch_size, n_caps_j, 1))), (tf.reshape(mean_f_fin, (batch_size, n_caps_j, mat_len)), tf.reshape(act_j_f_fin, (batch_size, n_caps_j, 1)))


def create_prim_conv3d_caps(inputs, channels, kernel_size, strides, name, padding='VALID', activation=None, mdim=4):
    mdim2 = mdim*mdim
    batch_size = tf.shape(inputs)[0]
    poses = tf.layers.conv3d(inputs=inputs, filters=channels * mdim2, kernel_size=kernel_size,
                             strides=strides, padding=padding, activation=activation, name=name+'_pose')

    _, d, h, w, _ = poses.get_shape().as_list()
    d, h, w = map(int, [d, h, w])

    pose = tf.reshape(poses, (batch_size, d, h, w, channels, mdim2), name=name+'_pose_res')
    #pose = tf.nn.l2_normalize(pose, dim=-1)

    acts = tf.layers.conv3d(inputs=inputs, filters=channels, kernel_size=kernel_size,
                             strides=strides, padding=padding, activation=tf.nn.sigmoid, name=name+'_act')
    activation = tf.reshape(acts, (batch_size, d, h, w, channels, 1), name=name+'_act_res')

    return pose, activation


def create_coords_mat(pose, rel_center, mdim=4):
    """

    :param pose: the incoming map of pose matrices, shape (N, ..., Ch_i, 16) where ... is the dimensions of the map can
    be 1, 2 or 3 dimensional.
    :param rel_center: whether or not the coordinates are relative to the center of the map
    :return: returns the coordinates (padded to 16) fir the incoming capsules
    """
    batch_size = tf.shape(pose)[0]
    shape_list = [int(x) for x in pose.get_shape().as_list()[1:-2]]
    ch = int(pose.get_shape().as_list()[-2])
    n_dims = len(shape_list)

    if n_dims == 3:
        d, h, w = shape_list
    elif n_dims == 2:
        d = 1
        h, w = shape_list
    else:
        d, h = 1, 1
        w = shape_list[0]

    subs = [0, 0, 0]
    if rel_center:
        subs = [int(d / 2), int(h / 2), int(w / 2)]

    c_mats = []
    if n_dims >= 3:
        c_mats.append(tf.tile(tf.reshape(tf.range(d, dtype=tf.float32), (1, d, 1, 1, 1, 1)), [batch_size, 1, h, w, ch, 1])-subs[0])
    if n_dims >= 2:
        c_mats.append(tf.tile(tf.reshape(tf.range(h, dtype=tf.float32), (1, 1, h, 1, 1, 1)), [batch_size, d, 1, w, ch, 1])-subs[1])
    if n_dims >= 1:
        c_mats.append(tf.tile(tf.reshape(tf.range(w, dtype=tf.float32), (1, 1, 1, w, 1, 1)), [batch_size, d, h, 1, ch, 1])-subs[2])
    add_coords = tf.concat(c_mats, axis=-1)
    add_coords = tf.cast(tf.reshape(add_coords, (batch_size, d*h*w, ch, n_dims)), dtype=tf.float32)

    mdim2 = mdim*mdim
    zeros = tf.zeros((batch_size, d*h*w, ch, mdim2-n_dims))

    return tf.concat([zeros, add_coords], axis=-1)


def create_dense_caps(inputs, n_caps_j, name, route_min=0.0, coord_add=False, rel_center=False,
                      ch_same_w=True, mdim=4):
    """

    :param inputs: The input capsule layer. Shape ((N, K, Ch_i, 16), (N, K, Ch_i, 1)) or
    ((N, ..., Ch_i, 16), (N, ..., Ch_i, 1)) where K is the number of capsules per channel and '...' is if you are
    inputting an map of capsules like W or HxW or DxHxW.
    :param n_caps_j: The number of capsules in the following layer
    :param name: name of the capsule layer
    :param route_min: A threshold activation to route
    :param coord_add: A boolean, whether or not to to do coordinate addition
    :param rel_center: A boolean, whether or not the coordinate addition is relative to the center
    :param routing_type: The type of routing
    :return: Returns a layer of capsules. Shape ((N, n_caps_j, 16), (N, n_caps_j, 1))
    """
    mdim2 = mdim*mdim
    pose, activation = inputs
    batch_size = tf.shape(pose)[0]
    shape_list = [int(x) for x in pose.get_shape().as_list()[1:]]
    ch = int(shape_list[-2])
    n_capsch_i = 1 if len(shape_list) == 2 else reduce((lambda x, y: x * y), shape_list[:-2])

    u_i = tf.reshape(pose, (batch_size, n_capsch_i, ch, mdim2))
    activation = tf.reshape(activation, (batch_size, n_capsch_i, ch, 1))
    coords = create_coords_mat(pose, rel_center) if coord_add else tf.zeros_like(u_i)

    # reshapes the input capsules
    u_i = tf.reshape(u_i, (batch_size, n_capsch_i, ch, mdim, mdim))
    u_i = tf.expand_dims(u_i, axis=-3)
    u_i = tf.tile(u_i, [1, 1, 1, n_caps_j, 1, 1])

    if ch_same_w:
        weights = tf.get_variable(name=name + '_weights', shape=(ch, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))

        votes = tf.einsum('ijab,ntijbc->ntijac', weights, u_i)
        votes = tf.reshape(votes, (batch_size, n_capsch_i * ch, n_caps_j, mdim2), name=name+'_votes')
    else:
        weights = tf.get_variable(name=name + '_weights', shape=(n_capsch_i, ch, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))
        votes = tf.einsum('tijab,ntijbc->ntijac', weights, u_i)
        votes = tf.reshape(votes, (batch_size, n_capsch_i * ch, n_caps_j, mdim2), name=name+'_votes')

    if coord_add:
        coords = tf.reshape(coords, (batch_size, n_capsch_i * ch, 1, mdim2))
        votes = votes + tf.tile(coords, [1, 1, n_caps_j, 1])

    acts = tf.reshape(activation, (batch_size, n_capsch_i * ch, 1))
    activations = tf.where(tf.greater_equal(acts, tf.constant(route_min)), acts, tf.zeros_like(acts))

    beta_v = tf.get_variable(name=name + '_beta_v', shape=(n_caps_j, mdim2),
                             initializer=tf.initializers.random_normal(stddev=0.1),
                             regularizer=tf.contrib.layers.l2_regularizer(0.1))

    beta_a = tf.get_variable(name=name + '_beta_a', shape=(n_caps_j, 1),
                             initializer=tf.initializers.random_normal(stddev=0.1),
                             regularizer=tf.contrib.layers.l2_regularizer(0.1))

    capsules = em_routing(votes, activations, beta_v, beta_a)

    return capsules


def create_conv3d_caps(inputs, channels, kernel_size, strides, name, padding='VALID',
                       coord_add=False, rel_center=True, route_mean=True, ch_same_w=True, mdim=4):
    mdim2 = mdim*mdim
    inputs = tf.concat(inputs, axis=-1)

    if padding == 'SAME':
        d_padding, h_padding, w_padding = int(float(kernel_size[0]) / 2), int(float(kernel_size[1]) / 2), int(float(kernel_size[2]) / 2)
        u_padded = tf.pad(inputs, [[0, 0], [d_padding, d_padding], [h_padding, h_padding], [w_padding, w_padding], [0, 0], [0, 0]])
    else:
        u_padded = inputs

    batch_size = tf.shape(u_padded)[0]
    _, d, h, w, ch, _ = u_padded.get_shape().as_list()
    d, h, w, ch = map(int, [d, h, w, ch])

    # gets indices for kernels
    d_offsets = [[(d_ + k) for k in range(kernel_size[0])] for d_ in range(0, d + 1 - kernel_size[0], strides[0])]
    h_offsets = [[(h_ + k) for k in range(kernel_size[1])] for h_ in range(0, h + 1 - kernel_size[1], strides[1])]
    w_offsets = [[(w_ + k) for k in range(kernel_size[2])] for w_ in range(0, w + 1 - kernel_size[2], strides[2])]

    # output dimensions
    d_out, h_out, w_out = len(d_offsets), len(h_offsets), len(w_offsets)

    # gathers the capsules into shape (N, D2, H2, W2, KD, KH, KW, Ch_in, 17)
    d_gathered = tf.gather(u_padded, d_offsets, axis=1)
    h_gathered = tf.gather(d_gathered, h_offsets, axis=3)
    w_gathered = tf.gather(h_gathered, w_offsets, axis=5)
    w_gathered = tf.transpose(w_gathered, [0, 1, 3, 5, 2, 4, 6, 7, 8])

    if route_mean:
        kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out, kernel_size[0]* kernel_size[1]* kernel_size[2], ch, mdim2 + 1])
        kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
        capsules = create_dense_caps((kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]), channels, name,
                                     ch_same_w=ch_same_w, mdim=mdim)
    else:
        kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out, kernel_size[0], kernel_size[1], kernel_size[2], ch, mdim2 + 1])
        capsules = create_dense_caps((kernels_reshaped[:, :, :, :, :, :-1], kernels_reshaped[:, :, :, :, :, -1:]),
                                     channels, name, coord_add=coord_add, rel_center=rel_center, ch_same_w=ch_same_w, mdim=mdim)

    poses = tf.reshape(capsules[0][:, :, :mdim2], (batch_size, d_out, h_out, w_out, channels, mdim2), name=name+'_pose')
    activations = tf.reshape(capsules[1], (batch_size, d_out, h_out, w_out, channels, 1), name=name+'_act')

    return poses, activations


def create_dense_caps_cond(inputs, n_caps_j, name, coord_add=False, rel_center=False,
                           ch_same_w=True, mdim=4, n_cond_caps=0):
    """

    :param inputs: The input capsule layer. Shape ((N, K, Ch_i, 16), (N, K, Ch_i, 1)) or
    ((N, ..., Ch_i, 16), (N, ..., Ch_i, 1)) where K is the number of capsules per channel and '...' is if you are
    inputting an map of capsules like W or HxW or DxHxW.
    :param n_caps_j: The number of capsules in the following layer
    :param name: name of the capsule layer
    :param coord_add: A boolean, whether or not to to do coordinate addition
    :param rel_center: A boolean, whether or not the coordinate addition is relative to the center
    :param routing_type: The type of routing
    :return: Returns a layer of capsules. Shape ((N, n_caps_j, 16), (N, n_caps_j, 1))
    """
    mdim2 = mdim*mdim
    pose, activation = inputs
    batch_size = tf.shape(pose)[0]
    shape_list = [int(x) for x in pose.get_shape().as_list()[1:]]
    ch = int(shape_list[-2])
    n_capsch_i = 1 if len(shape_list) == 2 else reduce((lambda x, y: x * y), shape_list[:-2])

    u_i = tf.reshape(pose, (batch_size, n_capsch_i, ch, mdim2))
    activation = tf.reshape(activation, (batch_size, n_capsch_i, ch, 1))
    coords = create_coords_mat(pose, rel_center) if coord_add else tf.zeros_like(u_i)


    # reshapes the input capsules
    u_i = tf.reshape(u_i, (batch_size, n_capsch_i, ch, mdim, mdim))
    u_i = tf.expand_dims(u_i, axis=-3)
    u_i = tf.tile(u_i, [1, 1, 1, n_caps_j, 1, 1])

    if ch_same_w:
        weights = tf.get_variable(name=name + '_weights', shape=(ch, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))

        votes = tf.einsum('ijab,ntijbc->ntijac', weights, u_i)
        votes = tf.reshape(votes, (batch_size, n_capsch_i * ch, n_caps_j, mdim2), name=name+'_votes')
    else:
        weights = tf.get_variable(name=name + '_weights', shape=(n_capsch_i, ch, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))
        votes = tf.einsum('tijab,ntijbc->ntijac', weights, u_i)
        votes = tf.reshape(votes, (batch_size, n_capsch_i * ch, n_caps_j, mdim2), name=name+'_votes')

    if coord_add:
        coords = tf.reshape(coords, (batch_size, n_capsch_i * ch, 1, mdim2))
        votes = votes + tf.tile(coords, [1, 1, n_caps_j, 1])


    if n_cond_caps == 0:
        beta_v = tf.get_variable(name=name + '_beta_v', shape=(n_caps_j, mdim2),
                                 initializer=tf.initializers.random_normal(stddev=0.1),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.1))
        beta_a = tf.get_variable(name=name + '_beta_a', shape=(n_caps_j, 1),
                                 initializer=tf.initializers.random_normal(stddev=0.1),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.1))

        acts = tf.reshape(activation, (batch_size, n_capsch_i * ch, 1))

        capsules1 = em_routing(votes, acts, beta_v, beta_a)
        capsules2 = capsules1
    else:
        beta_v1 = tf.get_variable(name=name + '_beta_v1', shape=(n_caps_j, mdim2),
                                 initializer=tf.initializers.random_normal(stddev=0.1),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.1))
        beta_a1 = tf.get_variable(name=name + '_beta_a1', shape=(n_caps_j, 1),
                                 initializer=tf.initializers.random_normal(stddev=0.1),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.1))

        beta_v2 = tf.get_variable(name=name + '_beta_v2', shape=(n_caps_j, mdim2),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))
        beta_a2 = tf.get_variable(name=name + '_beta_a2', shape=(n_caps_j, 1),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))

        votes = tf.reshape(votes, (batch_size, n_capsch_i, ch, n_caps_j, mdim2))

        votes1 = tf.reshape(votes[:, :, :ch - n_cond_caps],
                            (batch_size, n_capsch_i * (ch - n_cond_caps), n_caps_j, mdim2))
        votes2 = tf.reshape(votes[:, :, ch - n_cond_caps:], (batch_size, n_capsch_i * n_cond_caps, n_caps_j, mdim2))

        acts = tf.reshape(activation, (batch_size, n_capsch_i, ch, 1))

        acts1 = tf.reshape(acts[:, :, :ch - n_cond_caps], (batch_size, n_capsch_i * (ch - n_cond_caps), 1))
        acts2 = tf.reshape(acts[:, :, ch - n_cond_caps:], (batch_size, n_capsch_i * n_cond_caps, 1))

        weights_2 = tf.get_variable(name=name + '_weights_2', shape=(ch - n_cond_caps, n_caps_j, mdim, mdim),
                                  initializer=tf.initializers.random_normal(stddev=0.1),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.1))

        votes_2 = tf.einsum('ijab,ntijbc->ntijac', weights_2, u_i[:, :, :ch - n_cond_caps])
        votes_2 = tf.reshape(votes_2, (batch_size, n_capsch_i * (ch - n_cond_caps), n_caps_j, mdim2), name=name + '_votes2')

        capsules1, capsules2 = em_routing_cond(votes1, votes_2, acts1, votes2, acts2, beta_v1, beta_a1, beta_v2,
                                               beta_a2)

    return capsules1, capsules2


def create_conv3d_caps_cond(inputs, channels, kernel_size, strides, name, padding='VALID',
                            coord_add=False, rel_center=True, route_mean=True, ch_same_w=True, mdim=4, n_cond_caps=0):
    mdim2 = mdim*mdim
    inputs = tf.concat(inputs, axis=-1)

    if padding == 'SAME':
        d_padding, h_padding, w_padding = int(float(kernel_size[0]) / 2), int(float(kernel_size[1]) / 2), int(float(kernel_size[2]) / 2)
        u_padded = tf.pad(inputs, [[0, 0], [d_padding, d_padding], [h_padding, h_padding], [w_padding, w_padding], [0, 0], [0, 0]])
    else:
        u_padded = inputs

    batch_size = tf.shape(u_padded)[0]
    _, d, h, w, ch, _ = u_padded.get_shape().as_list()
    d, h, w, ch = map(int, [d, h, w, ch])

    # gets indices for kernels
    d_offsets = [[(d_ + k) for k in range(kernel_size[0])] for d_ in range(0, d + 1 - kernel_size[0], strides[0])]
    h_offsets = [[(h_ + k) for k in range(kernel_size[1])] for h_ in range(0, h + 1 - kernel_size[1], strides[1])]
    w_offsets = [[(w_ + k) for k in range(kernel_size[2])] for w_ in range(0, w + 1 - kernel_size[2], strides[2])]

    # output dimensions
    d_out, h_out, w_out = len(d_offsets), len(h_offsets), len(w_offsets)

    # gathers the capsules into shape (N, D2, H2, W2, KD, KH, KW, Ch_in, 17)
    d_gathered = tf.gather(u_padded, d_offsets, axis=1)
    h_gathered = tf.gather(d_gathered, h_offsets, axis=3)
    w_gathered = tf.gather(h_gathered, w_offsets, axis=5)
    w_gathered = tf.transpose(w_gathered, [0, 1, 3, 5, 2, 4, 6, 7, 8])

    if route_mean:
        kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out, kernel_size[0]* kernel_size[1]* kernel_size[2], ch, mdim2 + 1])
        kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
        capsules1, capsules2 = create_dense_caps_cond((kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]), channels, name,
                                           ch_same_w=ch_same_w, mdim=mdim, n_cond_caps=n_cond_caps)
    else:
        kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out, kernel_size[0], kernel_size[1], kernel_size[2], ch, mdim2 + 1])
        capsules1, capsules2 = create_dense_caps_cond((kernels_reshaped[:, :, :, :, :, :-1], kernels_reshaped[:, :, :, :, :, -1:]),
                                                      channels, name, coord_add=coord_add, rel_center=rel_center,
                                                      ch_same_w=ch_same_w, mdim=mdim, n_cond_caps=n_cond_caps)

    poses1 = tf.reshape(capsules1[0][:, :, :mdim2], (batch_size, d_out, h_out, w_out, channels, mdim2), name=name+'_pose1')
    activations1 = tf.reshape(capsules1[1], (batch_size, d_out, h_out, w_out, channels, 1), name=name+'_act1')

    poses2 = tf.reshape(capsules2[0][:, :, :mdim2], (batch_size, d_out, h_out, w_out, channels, mdim2), name=name + '_pose2')
    activations2 = tf.reshape(capsules2[1], (batch_size, d_out, h_out, w_out, channels, 1), name=name + '_act2')

    return (poses1, activations1), (poses2, activations2)


def layer_shape(layer):
    return str(layer[0].get_shape()) + ' ' + str(layer[1].get_shape())

