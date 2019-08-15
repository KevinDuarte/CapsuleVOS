import tensorflow as tf
import config
from caps_network_train import CapsNet
import sys
from load_youtube_data_multi import YoutubeTrainDataGen as TrainDataGen
from load_youtubevalid_data import YoutubeValidDataGen as ValidDataGen
import numpy as np
import time


def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Num of parameters:', total_parameters)
    sys.stdout.flush()


def train_one_epoch(sess, capsnet, data_gen, epoch):
    start_time = time.time()
    # continues until no more training data is generated
    batch, s_losses, seg_acc, reg_losses = 0.0, 0, 0, 0

    while data_gen.has_data():
        x_batch, seg_batch, crop1_batch, crop2_batch = data_gen.get_batch(config.batch_size)

        if config.multi_gpu and len(x_batch) == 1:
            print('Batch size of one, not running')
            continue

        n_samples = len(x_batch)

        use_gt_seg = epoch <= config.n_epochs_for_gt_seg
        use_gt_crop = epoch <= config.n_epochs_for_gt_crop

        hr_lstm_input = np.zeros((n_samples, config.hr_lstm_size[0], config.hr_lstm_size[1], config.hr_lstm_feats))
        lr_lstm_input = np.zeros((n_samples, config.lr_lstm_size[0], config.lr_lstm_size[1], config.lr_lstm_feats))

        outputs = sess.run([capsnet.train_op, capsnet.segmentation_loss, capsnet.pred_caps, capsnet.seg_acc,
                            capsnet.regression_loss],
                           feed_dict={capsnet.x_input_video: x_batch, capsnet.y_segmentation: seg_batch,
                                      capsnet.hr_cond_input: hr_lstm_input, capsnet.lr_cond_input: lr_lstm_input,
                                      capsnet.use_gt_seg: use_gt_seg, capsnet.use_gt_crop: use_gt_crop,
                                      capsnet.gt_crops1: crop1_batch, capsnet.gt_crops2: crop2_batch})

        _, s_loss, cap_vals, s_acc, reg_loss = outputs
        s_losses += s_loss
        seg_acc += s_acc
        reg_losses += reg_loss

        batch += 1

        if np.isnan(cap_vals[0][0]):
            print(cap_vals[0][:10])
            print('NAN encountered.')
            config.write_output('NAN encountered.\n')
            return -1, -1, -1

        if batch % config.batches_until_print == 0:
            print('Finished %d batches. %d(s) since start. Avg Segmentation Loss is %.4f. Avg Regression Loss is %.4f. '
                  'Seg Acc is %.4f.'
                  % (batch, time.time() - start_time, s_losses / batch, reg_losses / batch, seg_acc / batch))
            sys.stdout.flush()

    print('Finish Epoch in %d(s). Avg Segmentation Loss is %.4f. Avg Regression Loss is %.4f. Seg Acc is %.4f.' %
          (time.time() - start_time, s_losses / batch, reg_losses / batch, seg_acc / batch))
    sys.stdout.flush()

    return s_losses / batch, reg_losses / batch, seg_acc / batch


def validate(sess, capsnet, data_gen):
    batch, s_losses, seg_acc = 0.0, 0, 0
    start_time = time.time()

    while data_gen.has_data():
        batch_data = data_gen.get_batch(config.batch_size)
        x_batch, seg_batch, crop1_batch = batch_data

        hr_lstm_input = np.zeros((len(x_batch), config.hr_lstm_size[0], config.hr_lstm_size[1], config.hr_lstm_feats))
        lr_lstm_input = np.zeros((len(x_batch), config.lr_lstm_size[0], config.lr_lstm_size[1], config.lr_lstm_feats))

        val_ouputs = sess.run([capsnet.val_seg_loss, capsnet.val_seg_acc],
                              feed_dict={capsnet.x_input_video: x_batch, capsnet.y_segmentation: seg_batch,
                                         capsnet.hr_cond_input: hr_lstm_input, capsnet.lr_cond_input: lr_lstm_input,
                                         capsnet.use_gt_seg: True, capsnet.use_gt_crop: True,
                                         capsnet.gt_crops1: crop1_batch, capsnet.gt_crops2: crop1_batch})

        s_loss, s_acc = val_ouputs

        s_losses += s_loss
        seg_acc += s_acc

        batch += 1

        if batch % config.batches_until_print == 0:
            print('Tested %d batches. %d(s) since start.' % (batch, time.time() - start_time))
            sys.stdout.flush()

    print('Evaluation done in %d(s).' % (time.time() - start_time))
    print('Test Segmentation Loss: %.4f. Test Segmentation Acc: %.4f' % (s_losses / batch, seg_acc / batch))
    sys.stdout.flush()

    return s_losses / batch, seg_acc / batch


def train_network(gpu_config):
    capsnet = CapsNet()

    with tf.Session(graph=capsnet.graph, config=gpu_config) as sess:
        tf.global_variables_initializer().run()

        get_num_params()
        if config.start_at_epoch <= 1:
            config.clear_output()
        else:
            capsnet.load(sess, config.save_file_best_name % (config.start_at_epoch - 1))
            print('Loading from epoch %d.' % (config.start_at_epoch-1))

        best_loss = 1000000
        best_epoch = 1
        print('Training on YoutubeVOS')
        for ep in range(config.start_at_epoch, config.n_epochs + 1):
            print(20 * '*', 'epoch', ep, 20 * '*')
            sys.stdout.flush()

            # Trains network for 1 epoch
            nan_tries = 0
            while nan_tries < 3:
                data_gen = TrainDataGen(config.wait_for_data, crop_size=config.hr_frame_size, n_frames=config.n_frames,
                                        rand_frame_skip=config.rand_frame_skip, multi_objects=config.multiple_objects)
                seg_loss, reg_loss, seg_acc = train_one_epoch(sess, capsnet, data_gen, ep)

                if seg_loss < 0 or seg_acc < 0:
                    nan_tries += 1
                    capsnet.load(sess, config.save_file_best_name % best_epoch)  # loads in the previous epoch
                    while data_gen.has_data():
                        data_gen.get_batch(config.batch_size)
                else:
                    config.write_output('Epoch %d: SL: %.4f. RL: %.4f. SegAcc: %.4f.\n' % (ep, seg_loss, reg_loss, seg_acc))
                    break

            if nan_tries == 3:
                print('Network cannot be trained. Too many NaN issues.')
                exit()

            # Validates network
            data_gen = ValidDataGen(config.wait_for_data, crop_size=config.hr_frame_size, n_frames=config.n_frames)
            seg_loss, seg_acc = validate(sess, capsnet, data_gen)

            config.write_output('Validation\tSL: %.4f. SA: %.4f.\n' % (seg_loss, seg_acc))

            # saves every 10 epochs
            if ep % config.save_every_n_epochs == 0:
                try:
                    capsnet.save(sess, config.save_file_name % ep)
                    config.write_output('Saved Network\n')
                except:
                    print('Failed to save network!!!')
                    sys.stdout.flush()

            # saves when validation loss becomes smaller (after 50 epochs to save space)
            t_loss = seg_loss

            if t_loss < best_loss:
                best_loss = t_loss
                try:
                    capsnet.save(sess, config.save_file_best_name % ep)
                    best_epoch = ep
                    config.write_output('Saved Network - Minimum val\n')
                except:
                    print('Failed to save network!!!')
                    sys.stdout.flush()

    tf.reset_default_graph()


def main():
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True

    train_network(gpu_config)


sys.settrace(main())


