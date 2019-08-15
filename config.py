import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = ['/gpu:0', '/gpu:0']
multi_gpu = len(set(devices)) == 2

batch_size = 4
n_epochs = 200

learning_rate, beta1, epsilon = 0.0001, 0.5, 1e-6

n_frames = 8
n_epochs_for_gt_seg = 0
n_epochs_for_gt_crop = 500

hr_frame_size = (128*4, 224*4)
hr_lstm_size = (1, 1)
hr_lstm_feats = 256

lr_frame_size = (128, 224)
lr_lstm_size = (lr_frame_size[0]//4, lr_frame_size[1]//4)
lr_lstm_feats = 256

model_num = 4

save_every_n_epochs = 50
output_file_name = './output%d.txt' % model_num
save_file_name = ('./network_saves/model%d' % model_num) + '_%d.ckpt'
save_file_best_name = ('./network_saves_best/model%d' % model_num) + '_%d.ckpt'
start_at_epoch = 1

output_inference_file = './Anns2/Annotations/'
epoch_save = 394
save_file_inference = ('./network_saves_best/model%d' % model_num) + '_%d.ckpt'

multiple_objects = False
rand_frame_skip = 1
wait_for_data = 5  # in seconds
batches_until_print = 1

inv_temp = 0.5
inv_temp_delta = 0.1
pose_dimension = 4

print_layers = True


def clear_output():
    with open(output_file_name, 'w') as f:
        print('Writing to ' + output_file_name)
        f.write('Model #: %d. Batch Size: %d.\n' % (model_num, batch_size))


def write_output(string):
    try:
        output_log = open(output_file_name, 'a')
        output_log.write(string)
        output_log.close()
    except:
        print('Unable to save to output log')
