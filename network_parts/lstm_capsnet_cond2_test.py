import tensorflow as tf
from caps_layers_cond import create_prim_conv3d_caps, create_conv3d_caps, layer_shape, create_conv3d_caps_cond
import config


# Basic network:
# HR frame branch has LSTM right before prediction and bounding box of shape y1, x1, alpha (keeps 128x224 aspect ratio)
# LR frame branch has ConvLSTM at 1/4 LR input resolution
# Skip connections only from conditioned capsules and video capsules (no frame capsules given to decoder)

# Batch size = 3 on TitanXp

def conv_lstm_layer(x_t, c_tm1, h_tm1, n_feats, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='lstm_convs'):
    inp_cat = tf.concat([x_t, h_tm1], axis=-1)
    conv_outputs = tf.layers.conv2d(inp_cat, n_feats*4, kernel_size, strides, padding=padding, name=name)

    input_gate, new_input, forget_gate, output_gate = tf.split(conv_outputs, 4, axis=-1)

    forget_bias = 0.0  # changed this from 1.0

    c_t = tf.nn.sigmoid(forget_gate + forget_bias) * c_tm1
    c_t += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)  # tf.nn.tanh(new_input)
    h_t = tf.nn.tanh(c_t) * tf.sigmoid(output_gate)

    return c_t, h_t


def transposed_conv3d(inputs, n_units, kernel_size, strides, padding='VALID', activation=tf.nn.relu, name='deconv', use_bias=True):
    conv = tf.layers.conv3d_transpose(inputs, n_units, kernel_size=kernel_size, strides=strides, padding=padding,
                                      use_bias=False, activation=activation, name=name)
    if use_bias:
        bias = tf.get_variable(name=name + '_bias', shape=(1, 1, 1, 1, n_units))
        return activation(conv + bias)
    else:
        return activation(conv)


def create_skip_connection(in_caps_layer, n_units, kernel_size, strides=(1, 1, 1), padding='VALID', name='skip', activation=tf.nn.relu):
    in_caps_layer = in_caps_layer[0]
    batch_size = tf.shape(in_caps_layer)[0]
    _, d, h, w, ch, dim = in_caps_layer.get_shape()
    d, h, w, ch, dim = map(int, [d, h, w, ch, dim])

    in_caps_res = tf.reshape(in_caps_layer, [batch_size, d, h, w, ch * dim])

    return tf.layers.conv3d_transpose(in_caps_res, n_units, kernel_size=kernel_size, strides=strides, padding=padding,
                                      use_bias=False, activation=activation, name=name)


def video_encoder(x_input):
    x = tf.layers.conv3d(x_input, 32, kernel_size=[1, 3, 3], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv1_2d')
    x = tf.layers.conv3d(x, 64, kernel_size=[3, 1, 1], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv1_1d')

    x = tf.layers.conv3d(x, 64, kernel_size=[1, 3, 3], padding='SAME', strides=[1, 2, 2],
                         activation=tf.nn.relu, name='conv2_2d')
    x = tf.layers.conv3d(x, 128, kernel_size=[3, 1, 1], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv2_1d')

    x = tf.layers.conv3d(x, 256, kernel_size=[1, 3, 3], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv3a_2d')
    x = tf.layers.conv3d(x, 256, kernel_size=[3, 1, 1], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv3a_1d')
    x = tf.layers.conv3d(x, 256, kernel_size=[1, 3, 3], padding='SAME', strides=[1, 2, 2],
                         activation=tf.nn.relu, name='conv3b_2d')
    x = tf.layers.conv3d(x, 256, kernel_size=[3, 1, 1], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv3b_1d')

    x = tf.layers.conv3d(x, 512, kernel_size=[1, 3, 3], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv4a_2d')
    x = tf.layers.conv3d(x, 512, kernel_size=[3, 1, 1], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv4a_1d')
    x = tf.layers.conv3d(x, 512, kernel_size=[1, 3, 3], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv4b_2d')
    x = tf.layers.conv3d(x, 512, kernel_size=[3, 1, 1], padding='SAME', strides=[1, 1, 1],
                         activation=tf.nn.relu, name='conv4b_1d')

    return x


def lr_frame_encoder(frame_plus_seg, c_tm1_lr, h_tm1_lr):
    fr_conv1 = tf.layers.conv2d(frame_plus_seg, 32, kernel_size=[3, 3], padding='SAME', strides=[1, 1],
                                activation=tf.nn.relu, name='lr_fr_conv1')
    fr_conv2 = tf.layers.conv2d(fr_conv1, 64, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='lr_fr_conv2')

    fr_conv3 = tf.layers.conv2d(fr_conv2, 64, kernel_size=[3, 3], padding='SAME', strides=[1, 1],
                                activation=tf.nn.relu, name='lr_fr_conv3')
    fr_conv4 = tf.layers.conv2d(fr_conv3, 128, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='lr_fr_conv4')

    fr_conv5 = tf.layers.conv2d(fr_conv4, config.lr_lstm_feats // 2, kernel_size=[3, 3], padding='SAME', strides=[1, 1],
                                activation=tf.nn.relu, name='lr_fr_conv5')
    c_t, h_t = conv_lstm_layer(fr_conv5, c_tm1_lr, h_tm1_lr, config.lr_lstm_feats // 2, name='lr_fr_lstm')

    return c_t, h_t


def hr_frame_encoder(frame_plus_seg, c_tm1, h_tm1):
    fr_conv1 = tf.layers.conv2d(frame_plus_seg, 32, kernel_size=[3, 3], padding='SAME', strides=[1, 1],
                                activation=tf.nn.relu, name='hr_fr_conv1')
    fr_conv2 = tf.layers.conv2d(fr_conv1, 64, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='hr_fr_conv2')  # 256, 448

    fr_conv3 = tf.layers.conv2d(fr_conv2, 128, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='hr_fr_conv3')  # 128, 224
    fr_conv4 = tf.layers.conv2d(fr_conv3, 128, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='hr_fr_conv4')  # 64, 112

    fr_conv5 = tf.layers.conv2d(fr_conv4, 256, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='hr_fr_conv5')  # 32, 56
    fr_conv6 = tf.layers.conv2d(fr_conv5, 256, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='hr_fr_conv6')  # 16, 28

    fr_conv7 = tf.layers.conv2d(fr_conv6, 512, kernel_size=[3, 3], padding='SAME', strides=[2, 2],
                                activation=tf.nn.relu, name='hr_fr_conv7')  # 8, 14

    fr_conv8 = tf.layers.conv2d(fr_conv7, config.hr_lstm_feats // 2, kernel_size=[8, 14], padding='VALID',
                                strides=[1, 1], activation=tf.nn.relu, name='hr_fr_conv8')  # 1, 1

    c_t, h_t = conv_lstm_layer(fr_conv8, c_tm1, h_tm1, config.hr_lstm_feats // 2, kernel_size=(1, 1))

    crop_pred = tf.layers.conv2d(h_t, 3, kernel_size=[1, 1], padding='VALID', strides=[1, 1],
                                 activation=tf.nn.sigmoid, name='crop_reg')

    y1, x1, alpha = tf.split(crop_pred[:, 0, 0, :], 3, axis=-1)

    alpha = 0.875 * alpha + 0.125
    zero = tf.zeros_like(alpha)

    return c_t, h_t, tf.concat((y1, x1, alpha, zero), axis=-1)


def convert_crop(crop, use_gt):
    y1, x1, alpha, _ = tf.split(crop, 4, axis=-1)
    y1 = tf.cond(use_gt, lambda: y1, lambda: y1)
    x1 = tf.cond(use_gt, lambda: x1, lambda: x1)
    alpha = tf.cond(use_gt, lambda: alpha, lambda: alpha)
    y1, x1, alpha = y1 - 0.05, x1 - 0.05, alpha + 0.1
    #y1, x1, alpha = y1 - 0.1, x1 - 0.1, alpha + 0.2

    y2 = y1+alpha
    x2 = x1+alpha

    crop_new = tf.concat((y1, x1, y2, x2), axis=-1)
    crop_new = tf.clip_by_value(crop_new, 0, 1)

    return crop_new


def uncrop(seg_out, crop_used):
    hr_h, hr_w = config.hr_frame_size

    y1, x1, y2, x2 = tf.split(crop_used, 4, axis=-1)

    up_padding = tf.cast(tf.floor(y1 * hr_h), tf.int32)
    left_padding = tf.cast(tf.floor(x1 * hr_w), tf.int32)

    img_res_h = tf.cast(tf.ceil((y2 - y1) * hr_h), tf.int32)
    img_res_w = tf.cast(tf.ceil((x2 - x1) * hr_w), tf.int32)

    down_padding = hr_h - img_res_h - up_padding
    right_padding = hr_w - img_res_w - left_padding

    fin_seg_init = tf.TensorArray(tf.float32, size=tf.shape(seg_out)[0])

    def cond(fin_seg, counter):
        return tf.less(counter, tf.shape(seg_out)[0])

    def res_and_pad(fin_seg, counter):
        segmentation = seg_out[counter]

        res_img = tf.image.resize_images(segmentation, (img_res_h[counter, 0], img_res_w[counter, 0]))

        padded_img = tf.pad(res_img, [[up_padding[counter, 0], down_padding[counter, 0]],
                                      [left_padding[counter, 0], right_padding[counter, 0]],
                                      [0, 0]], constant_values=-1000)

        fin_seg = fin_seg.write(counter, padded_img)

        return fin_seg, counter+1

    fin_seg, _ = tf.while_loop(cond, res_and_pad, [fin_seg_init, 0])

    return fin_seg.stack()


def create_decoder_network(pred_caps, sec_caps, prim_caps, print_layers=True):
    deconv1 = create_skip_connection(pred_caps, 128, kernel_size=[3, 3, 3], strides=[2, 2, 2], padding='SAME',
                                     name='deconv1')

    skip_connection1 = create_skip_connection(sec_caps, 128, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='SAME',
                                              name='skip_1')
    deconv1 = tf.concat([deconv1, skip_connection1], axis=-1)

    deconv2 = transposed_conv3d(deconv1, 128, kernel_size=[3, 3, 3], strides=[2, 2, 2], padding='SAME',
                                activation=tf.nn.relu, name='deconv2')

    skip_connection2 = create_skip_connection(prim_caps, 128, kernel_size=[1, 3, 3],
                                              strides=[1, 1, 1], padding='SAME', name='skip_2')
    deconv2 = tf.concat([deconv2, skip_connection2], axis=-1)

    deconv3 = transposed_conv3d(deconv2, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                activation=tf.nn.relu, name='deconv3')
    deconv4 = transposed_conv3d(deconv3, 256, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                activation=tf.nn.relu, name='deconv4')
    deconv5 = transposed_conv3d(deconv4, 128, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                activation=tf.nn.relu, name='deconv5')

    segment_layer = tf.layers.conv3d(deconv5, 1, kernel_size=[1, 1, 1], strides=[1, 1, 1], padding='SAME',
                                     activation=None, name='segment_layer')
    #segment_layer_sig = tf.nn.sigmoid(segment_layer)

    if print_layers:
        print('Deconv Layer 1:', deconv1.get_shape())
        print('Deconv Layer 2:', deconv2.get_shape())
        print('Deconv Layer 3:', deconv3.get_shape())
        print('Deconv Layer 4:', deconv4.get_shape())
        print('Deconv Layer 5:', deconv5.get_shape())
        print('Segment Layer:', segment_layer.get_shape())

    return segment_layer


def create_network_one_pass(hr_frames, hr_first_frame_seg, c_tm1, h_tm1, c_tm1_lr, h_tm1_lr, use_gt, gt_crop,
                            coord_addition, print_layers=True):
    # encodes hr frame, and gets the predicted crop
    hr_frame_plus_seg = tf.concat([hr_frames[:, 0], hr_first_frame_seg], axis=-1)
    c_t, h_t, pred_crop = hr_frame_encoder(hr_frame_plus_seg, c_tm1, h_tm1)

    center_y, center_x, alpha_gt, _ = tf.split(gt_crop, 4, axis=-1)
    _, _, alpha, _ = tf.split(pred_crop, 4, -1)

    y0 = tf.clip_by_value(center_y - 0.5*alpha, 0, 1)
    x0 = tf.clip_by_value(center_x - 0.5 * alpha, 0, 1)
    y1 = tf.clip_by_value(center_y - 0.5 * alpha_gt, 0, 1)
    x1 = tf.clip_by_value(center_x - 0.5 * alpha_gt, 0, 1)

    crop_to_use = tf.cond(use_gt, lambda:tf.concat([y1, x1, alpha_gt, alpha_gt], axis=-1), lambda: tf.concat([y0, x0, alpha, alpha], axis=-1))
    # crop_to_use = tf.cond(use_gt, lambda: gt_crop, lambda: pred_crop)

    # crop_to_use = tf.cond(use_gt, lambda: gt_crop, lambda: pred_crop)
    crop_to_use = convert_crop(crop_to_use, use_gt)

    frame_h, frame_w = config.lr_frame_size

    # crops the low resolution frame+seg
    range_crops = tf.range(tf.shape(crop_to_use)[0])
    lr_frame_plus_seg = tf.image.crop_and_resize(hr_frame_plus_seg, crop_to_use, range_crops, (frame_h, frame_w))

    c_t_lr, h_t_lr = lr_frame_encoder(lr_frame_plus_seg, c_tm1_lr, h_tm1_lr)

    # crops the video
    tiled_crop_to_use = tf.reshape(tf.tile(tf.expand_dims(crop_to_use, 1), [1, config.n_frames, 1]), (-1, 4))
    video_res = tf.reshape(hr_frames, (-1, config.hr_frame_size[0], config.hr_frame_size[1], 3))
    cropped_video = tf.image.crop_and_resize(video_res, tiled_crop_to_use, tf.range(tf.shape(tiled_crop_to_use)[0]),
                                             (frame_h, frame_w))
    cropped_video = tf.reshape(cropped_video, (-1, config.n_frames, frame_h, frame_w, 3))

    # creates video capsules
    lr_video_encoding = video_encoder(cropped_video)

    vid_caps = create_prim_conv3d_caps(lr_video_encoding, 12, kernel_size=[1, 3, 3], strides=[1, 2, 2], padding='SAME',
                                       name='vid_caps')

    vid_caps2 = vid_caps
    # vid_caps2 = (vid_caps[0] + coord_addition, vid_caps[1])

    # creates frame capsules, tiles them, and performs coordinate addition
    frame_caps = create_prim_conv3d_caps(tf.expand_dims(h_t_lr, axis=1), 8, kernel_size=[1, 3, 3],
                                         strides=[1, 2, 2], padding='SAME', name='frame_caps')
    frame_caps = (tf.tile(frame_caps[0], [1, config.n_frames, 1, 1, 1, 1]),
                  tf.tile(frame_caps[1], [1, config.n_frames, 1, 1, 1, 1]))

    frame_caps = (frame_caps[0] + coord_addition, frame_caps[1])

    # merges video and frame capsules
    prim_caps = (tf.concat([vid_caps2[0], frame_caps[0]], axis=-2), tf.concat([vid_caps2[1], frame_caps[1]], axis=-2))

    # performs capsule routing
    sec_caps, _ = create_conv3d_caps_cond(prim_caps, 16, kernel_size=[3, 3, 3], strides=[2, 2, 2], padding='SAME',
                                          name='sec_caps', route_mean=True, n_cond_caps=8)
    pred_caps = create_conv3d_caps(sec_caps, 16, kernel_size=[3, 3, 3], strides=[2, 2, 2], padding='SAME',
                                   name='third_caps', route_mean=True)
    fin_caps = tf.reduce_mean(pred_caps[1], [2, 3, 4, 5])

    if print_layers:
        print('Primary Caps:', layer_shape(prim_caps))
        print('Secondary Caps:', layer_shape(sec_caps))
        print('Prediction Caps:', layer_shape(pred_caps))

    # obtains the segmentations
    seg = create_decoder_network(pred_caps, sec_caps, vid_caps, print_layers=print_layers)

    hr_seg_out = uncrop(tf.reshape(seg, (-1, frame_h, frame_w, 1)), tiled_crop_to_use)
    hr_seg = tf.reshape(hr_seg_out, (-1, config.n_frames, config.hr_frame_size[0], config.hr_frame_size[1], 1))
    hr_seg_sig = tf.nn.sigmoid(hr_seg)

    return hr_seg, hr_seg_sig, c_t, h_t, c_t_lr, h_t_lr, pred_crop, fin_caps


def create_network(x_input, y_segmentation, f_tm1, f_tm1_lr, use_gt_crop, gt_crop1):
    coords_to_add = tf.reshape(tf.range(config.n_frames, dtype=tf.float32) / (config.n_frames - 1),
                               (1, config.n_frames, 1, 1, 1, 1))
    zeros_to_add = tf.zeros((1, config.n_frames, 1, 1, 1, 15), dtype=tf.float32)
    coords_to_add = tf.concat((zeros_to_add, coords_to_add), axis=-1)

    c_tm1, h_tm1 = f_tm1[:, :, :, :config.hr_lstm_feats // 2], f_tm1[:, :, :, config.hr_lstm_feats // 2:]
    c_tm1_lr, h_tm1_lr = f_tm1_lr[:, :, :, :config.lr_lstm_feats // 2], f_tm1_lr[:, :, :, config.lr_lstm_feats // 2:]

    network_outs1 = create_network_one_pass(x_input[:, :config.n_frames], y_segmentation, c_tm1, h_tm1, c_tm1_lr,
                                            h_tm1_lr, use_gt_crop, gt_crop1, coords_to_add, print_layers=True)

    seg, seg_sig, c_t, h_t, c_t_lr, h_t_lr, pred_crop1, fin_caps = network_outs1

    return seg, seg_sig, fin_caps, tf.concat([c_t, h_t], axis=-1), tf.concat([c_t_lr, h_t_lr], axis=-1), pred_crop1
