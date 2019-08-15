import config
import tensorflow as tf
import sys
from network_parts.lstm_capsnet_cond2_train import create_network


class CapsNet(object):
    def __init__(self, reg_mult=1.0):
        self.graph = tf.Graph()

        with self.graph.as_default():
            hr_h, hr_w = config.hr_frame_size

            n_frames = config.n_frames*2-1

            self.x_input_video = tf.placeholder(dtype=tf.float32, shape=(None, n_frames, hr_h, hr_w, 3),
                                                name='x_input_video')
            self.y_segmentation = tf.placeholder(dtype=tf.float32, shape=(None, n_frames, hr_h, hr_w, 1),
                                                 name='y_segmentation')

            self.use_gt_seg = tf.placeholder(dtype=tf.bool)
            self.use_gt_crop = tf.placeholder(dtype=tf.bool)

            self.gt_crops1 = tf.placeholder(dtype=tf.float32, shape=(None, 4))  # [y1, x1, y2, x2] between 0 and 1
            self.gt_crops2 = tf.placeholder(dtype=tf.float32, shape=(None, 4))  # [y1, x1, y2, x2] between 0 and 1

            cond_h, cond_w = config.hr_lstm_size
            self.hr_cond_input = tf.placeholder(dtype=tf.float32, shape=(None, cond_h, cond_w, config.hr_lstm_feats),
                                                name='hr_lstm_input')
            cond_h, cond_w = config.lr_lstm_size
            self.lr_cond_input = tf.placeholder(dtype=tf.float32, shape=(None, cond_h, cond_w, config.lr_lstm_feats),
                                                name='lr_lstm_input')
            self.init_network()

            self.init_seg_loss()
            self.init_regression_loss()
            self.total_loss = self.segmentation_loss + self.regression_loss*reg_mult

            self.init_optimizer()

            self.saver = tf.train.Saver()

    def init_network(self):
        print('Building Caps3d Model')

        with tf.variable_scope('network') as scope:
            if config.multi_gpu:
                b = tf.cast(tf.shape(self.x_input_video)[0] / 2, tf.int32)
                with tf.device(config.devices[0]):
                    segment_layer1, segment_layer_sig1, prim_caps1, state_t1, pred_crops11, pred_crops21 = create_network(self.x_input_video[:b],
                                                                                                                          self.y_segmentation[:b],
                                                                                                                          self.hr_cond_input[:b],
                                                                                                                          self.lr_cond_input[:b],
                                                                                                                          self.use_gt_seg,
                                                                                                                          self.use_gt_crop,
                                                                                                                          self.gt_crops1[:b],
                                                                                                                          self.gt_crops2[:b])

                scope.reuse_variables()
                with tf.device(config.devices[1]):
                    segment_layer2, segment_layer_sig2, prim_caps2, state_t2, pred_crops12, pred_crops22 = create_network(self.x_input_video[b:],
                                                                                                                          self.y_segmentation[b:],
                                                                                                                          self.hr_cond_input[b:],
                                                                                                                          self.lr_cond_input[b:],
                                                                                                                          self.use_gt_seg,
                                                                                                                          self.use_gt_crop,
                                                                                                                          self.gt_crops1[b:],
                                                                                                                          self.gt_crops2[b:])

                self.segment_layer = tf.concat([segment_layer1, segment_layer2], axis=0)
                self.segment_layer_sig = tf.concat([segment_layer_sig1, segment_layer_sig2], axis=0)
                self.pred_caps = tf.concat([prim_caps1, prim_caps2], axis=0)
                self.state_t = tf.concat([state_t1, state_t2], axis=0)
                self.pred_crops1 = tf.concat([pred_crops11, pred_crops12], axis=0)
                self.pred_crops2 = tf.concat([pred_crops21, pred_crops22], axis=0)
            else:
                network_outputs = create_network(self.x_input_video, self.y_segmentation, self.hr_cond_input, self.lr_cond_input, self.use_gt_seg, self.use_gt_crop, self.gt_crops1, self.gt_crops2)
                self.segment_layer, self.segment_layer_sig, self.pred_caps, self.state_t, self.pred_crops1, self.pred_crops2 = network_outputs

        sys.stdout.flush()

    def init_seg_loss(self):
        # Segmentation loss
        segment = self.segment_layer
        y_seg = self.y_segmentation

        segmentation_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_seg, logits=segment)
        #segmentation_loss = tf.reduce_mean(tf.reduce_sum(segmentation_loss, axis=[1, 2, 3, 4]))
        segmentation_loss = tf.reduce_mean(tf.reduce_mean(segmentation_loss, axis=[1, 2, 3, 4]))

        pred_seg = tf.cast(tf.greater(segment, 0.0), tf.float32)
        seg_acc = tf.reduce_mean(tf.cast(tf.equal(pred_seg, y_seg), tf.float32))

        frame_segment = self.segment_layer[:, 0, :, :, :]
        y_frame_segment = self.y_segmentation[:, 0, :, :, :]

        val_seg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_frame_segment, logits=frame_segment)
        val_seg_loss = tf.reduce_mean(tf.reduce_sum(val_seg_loss, axis=[1, 2, 3]))

        val_pred_seg = tf.cast(tf.greater(frame_segment, 0.0), tf.float32)
        val_seg_acc = tf.reduce_mean(tf.cast(tf.equal(val_pred_seg, y_frame_segment), tf.float32))

        segment_sig = self.segment_layer_sig

        p_times_r = segment_sig*y_seg
        p_plus_r = segment_sig + y_seg
        inv_p_times_r = (1-segment_sig) * (1-y_seg)
        inv_p_plus_r = 2-segment_sig-y_seg
        eps = 1e-8

        term1 = (tf.reduce_sum(p_times_r, axis=[1, 2, 3])+eps)/(tf.reduce_sum(p_plus_r, axis=[1, 2, 3])+eps)
        term2 = (tf.reduce_sum(inv_p_times_r, axis=[1, 2, 3])+eps)/(tf.reduce_sum(inv_p_plus_r, axis=[1, 2, 3])+eps)

        dice_loss = tf.reduce_mean(1 - term1 - term2)

        self.segmentation_loss = segmentation_loss + dice_loss
        # self.segmentation_loss = segmentation_loss
        self.val_seg_loss = val_seg_loss

        self.seg_acc = seg_acc
        self.val_seg_acc = val_seg_acc


        print('Segmentation Loss Initialized')

    def init_regression_loss(self):
        regression_loss = tf.square(self.gt_crops1 - self.pred_crops1) + tf.square(self.gt_crops2 - self.pred_crops2)
        self.regression_loss = tf.reduce_mean(tf.reduce_sum(regression_loss, axis=1))

        print('Regression Loss Initialized')

    def init_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, name='Adam',
                                           epsilon=config.epsilon)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(loss=self.total_loss, colocate_gradients_with_ops=True)

    def save(self, sess, file_name):
        save_path = self.saver.save(sess, file_name)
        print("Model saved in file: %s" % save_path)
        sys.stdout.flush()

    def load(self, sess, file_name):
        self.saver.restore(sess, file_name)
        print('Model restored.')
        sys.stdout.flush()

