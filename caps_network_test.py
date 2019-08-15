import config
import tensorflow as tf
import sys
from network_parts.lstm_capsnet_cond2_test import create_network


class CapsNet(object):
    def __init__(self, graph=None):
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        with self.graph.as_default():
            hr_h, hr_w = config.hr_frame_size

            n_frames = config.n_frames

            self.x_input_video = tf.placeholder(dtype=tf.float32, shape=(None, n_frames, hr_h, hr_w, 3),
                                                name='x_input_video')
            self.y_segmentation = tf.placeholder(dtype=tf.float32, shape=(None, n_frames, hr_h, hr_w, 1),
                                                 name='y_segmentation')

            self.x_first_seg = tf.placeholder(dtype=tf.float32, shape=(None, hr_h, hr_w, 1), name='x_first_seg')

            self.use_gt_crop = tf.placeholder(dtype=tf.bool)

            self.gt_crops1 = tf.placeholder(dtype=tf.float32, shape=(None, 4))  # [y1, x1, alpha, 0] between 0 and 1

            cond_h, cond_w = config.hr_lstm_size
            self.hr_cond_input = tf.placeholder(dtype=tf.float32, shape=(None, cond_h, cond_w, config.hr_lstm_feats),
                                                name='hr_lstm_input')
            cond_h, cond_w = config.lr_lstm_size
            self.lr_cond_input = tf.placeholder(dtype=tf.float32, shape=(None, cond_h, cond_w, config.lr_lstm_feats),
                                                name='lr_lstm_input')

            self.init_network()

            #self.init_seg_loss()
            #self.init_regression_loss()
            #self.total_loss = self.segmentation_loss + self.regression_loss

            #self.init_optimizer()

            self.saver = tf.train.Saver()

    def init_network(self):
        print('Building Caps3d Model')

        with tf.variable_scope('network') as scope:
            #scope.reuse_variables()
            network_outputs = create_network(self.x_input_video, self.x_first_seg, self.hr_cond_input, self.lr_cond_input, self.use_gt_crop, self.gt_crops1)
            self.segment_layer, self.segment_layer_sig, self.pred_caps, self.state_t, self.state_t_lr, self.pred_crops1 = network_outputs

        sys.stdout.flush()

    def init_seg_loss(self):
        # Segmentation loss
        segment = self.segment_layer
        y_seg = self.segment_layer#self.y_segmentation

        segmentation_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_seg, logits=segment)
        segmentation_loss = tf.reduce_mean(tf.reduce_sum(segmentation_loss, axis=[1, 2, 3, 4]))

        pred_seg = tf.cast(tf.greater(segment, 0.0), tf.float32)
        seg_acc = tf.reduce_mean(tf.cast(tf.equal(pred_seg, y_seg), tf.float32))

        frame_segment = self.segment_layer[:, 0, :, :, :]
        y_frame_segment = self.segment_layer[:, 0, :, :, :]#self.y_segmentation[:, 0, :, :, :]

        val_seg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_frame_segment, logits=frame_segment)
        val_seg_loss = tf.reduce_mean(tf.reduce_sum(val_seg_loss, axis=[1, 2, 3]))

        val_pred_seg = tf.cast(tf.greater(frame_segment, 0.0), tf.float32)
        val_seg_acc = tf.reduce_mean(tf.cast(tf.equal(val_pred_seg, y_frame_segment), tf.float32))

        self.segmentation_loss = segmentation_loss
        self.val_seg_loss = val_seg_loss

        self.seg_acc = seg_acc
        self.val_seg_acc = val_seg_acc

        print('Segmentation Loss Initialized')

    def init_regression_loss(self):
        regression_loss = tf.square(self.gt_crops1 - self.pred_crops1)
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

