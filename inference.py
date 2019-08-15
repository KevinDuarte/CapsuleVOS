import tensorflow as tf
from caps_network_test import CapsNet
from skvideo.io import vread
import numpy as np
from PIL import Image
from scipy.misc import imresize
import os
import config
import time


def load_video(video_name):
    video = vread(video_name)
    
    t, h, w, _ = video.shape
    
    resized_video = []
    for frame in video:
        resized_video.append(imresize(frame, config.hr_frame_size))
    
    resized_video = np.stack(resized_video, axis=0)
    
    return resized_video / 255., (h, w)


def load_first_frame(frame_name):
    image = Image.open(frame_name)
    palette = image.getpalette()
    image_np = np.array(image)
    
    return imresize(image_np, config.hr_frame_size, interp='nearest'), palette


def process_first_frame(first_frame):
    unique_seg_colors = np.unique(first_frame)

    fin_segmentations = {}
    for color in unique_seg_colors:
        if color == 0:
            continue
    
        gt_seg = np.where(first_frame == color, 1, 0)
        if np.sum(gt_seg) == 0:
                continue
        gt_seg = np.expand_dims(gt_seg, axis=-1)
        fin_segmentations[color] = (0, gt_seg)

    return fin_segmentations


def get_bounds(img):
    h_sum = np.sum(img, axis=1)
    w_sum = np.sum(img, axis=0)

    hs = np.where(h_sum > 0)
    ws = np.where(w_sum > 0)

    try:
        h0 = hs[0][0]
        h1 = hs[0][-1]
        w0 = ws[0][0]
        w1 = ws[0][-1]
    except:
        return -1, -1, -1, -1

    return h0, h1, w0, w1


def get_crop_to_use(h0, h1, w0, w1, h, w, prev_crop):
    # uses the h, w of predicted, and the center of gt
    if h0 == -1:
        use_gt_crop = False
        crop_to_use = prev_crop
    else:
        use_gt_crop = False
        crop_to_use = np.clip(np.array([((h0+h1)/2)/h, ((w0+w1)/2)/w, 1.0, 0]), 0, 1)

    return use_gt_crop, crop_to_use
    

def get_seg_for_clip_gt(sess, capsnet, clip, frame_start, lstm_cond, lstm_cond_lr, prev_crop):
    f, h, w, _ = clip.shape

    # print(frame_start.min())
    # print(frame_start.max())
    first_frame_seg_full = frame_start  # np.round(frame_start) # frame_start  #

    new_video_in = clip
    len_clip = new_video_in.shape[0]
    if len_clip < config.n_frames:
        new_video_in = np.concatenate((new_video_in, np.tile(new_video_in[-1:],
                                                             [config.n_frames - len_clip, 1, 1, 1])), axis=0)

    # gets the bounds of the segmentations
    h0, h1, w0, w1 = get_bounds(np.round(first_frame_seg_full[:, :, 0]))

    use_gt_crop, crop_to_use = get_crop_to_use(h0, h1, w0, w1, h, w, prev_crop)

    # runs through the network
    seg_pred, lstm_cond, lstm_cond_lr, pred_crops = sess.run([capsnet.segment_layer_sig, capsnet.state_t, capsnet.state_t_lr, capsnet.pred_crops1],
                                   feed_dict={capsnet.x_input_video: [new_video_in],
                                              capsnet.x_first_seg: [first_frame_seg_full],
                                              capsnet.hr_cond_input: lstm_cond,
                                              capsnet.lr_cond_input: lstm_cond_lr,
                                              capsnet.use_gt_crop: use_gt_crop,
                                              capsnet.gt_crops1: [crop_to_use]})

    # resizes crop and places it back into original frame size
    seg_pred = seg_pred[0]

    overlap_frames = 3

    if use_gt_crop:
        crop_to_use = crop_to_use
    else:
        crop_to_use = np.concatenate((crop_to_use[:2], pred_crops[0][2:]), axis=-1)

    return seg_pred, lstm_cond, lstm_cond_lr, overlap_frames, crop_to_use
    

def generate_inference(sess, capsnet, video, segmentations, orig_dim, vid_name, img_palette):
    orig_h, orig_w = orig_dim
    n_objects = int(max(segmentations.keys()))
    
    lstm_conds = np.zeros((n_objects + 1, config.hr_lstm_size[0], config.hr_lstm_size[1], config.hr_lstm_feats))
    lstm_conds_lr = np.zeros((n_objects + 1, config.lr_lstm_size[0], config.lr_lstm_size[1], config.lr_lstm_feats))

    prev_coords = np.zeros((n_objects + 1, 4))

    f, h, w, _ = video.shape

    segmentation_maps = np.zeros((config.n_frames, h, w, n_objects + 1))
    segmentation_maps[:, :, :, 0] = 0.5
    final_segmentation = np.zeros((h, w, 1))
    cur_i = np.ones((n_objects + 1,), np.uint8)
    overlaps = np.ones((n_objects + 1,), np.uint8)

    vid_dir = 'Output/' + vid_name + '/'
    mkdir(vid_dir)

    for i in range(f):
        for color in range(1, n_objects + 1):
            if color not in segmentations.keys():
                continue

            cur_i[color] += 1
            start_frame, start_seg = segmentations[color]

            if i < start_frame:  # the current frame occurs before the object appears
                cur_i[color] = 7
                continue
            elif i == start_frame:  # the current frame is the first frame of the object (use given segmentation)
                segmentation_maps[-1, :, :, color:color + 1] = start_seg
                cur_i[color] = 7
                continue

            if cur_i[color] != config.n_frames - overlaps[color] + 1:  # the current frame's segmentation has been predicted
                # cur_overlap[color] -= 1
                # segmentation_maps[:-1, :, :, color] = segmentation_maps[1:, :, :, color]
                continue

            cur_0 = cur_i[color] - 1

            # cond_frame_seg = segmentation_maps[i-1, :, :, color:color+1]  # This is the naive approach
            # cond_frame_seg = (final_segmentation[i-1] == color).astype(np.float32)  # winner take all approach
            cond_frame_seg = ((final_segmentation == color).astype(np.float32) + (final_segmentation == 0).astype(
                np.float32)) * segmentation_maps[cur_0, :, :, color:color + 1]  # winner take all approach 2

            vid_to_use = video[i - 1:i + config.n_frames - 1]
            orig_len = vid_to_use.shape[0]
            if orig_len < config.n_frames:
                vid_to_use = np.concatenate(
                    [vid_to_use] + [vid_to_use[-1:] for reps in range(config.n_frames - orig_len)], axis=0)

            # use previous frame to generate segmentation for future N frames
            pred_seg, lstm_cond, lstm_cond_lr, overlap_frames, coords_used = get_seg_for_clip_gt(sess, capsnet,
                                                                                                 vid_to_use,
                                                                                                 cond_frame_seg,
                                                                                                 lstm_conds[
                                                                                                 color:color + 1],
                                                                                                 lstm_conds_lr[
                                                                                                 color:color + 1],
                                                                                                 prev_coords[color])

            segmentation_maps[:, :, :, color:color + 1] = pred_seg
            lstm_conds[color:color + 1] = lstm_cond
            lstm_conds_lr[color:color + 1] = lstm_cond_lr
            overlaps[color] = overlap_frames

            prev_coords[color] = coords_used
            # print(overlap_frames)

            cur_i[color] = 1

        final_segmentation[:, :, 0] = np.argmax(segmentation_maps[cur_i, :, :, range(n_objects + 1)], axis=0)

        frame_name = vid_dir + ('%d.png' % i).zfill(5)

        fb_segs_argmax = imresize(final_segmentation[:, :, 0].astype(dtype=np.uint8), (orig_h, orig_w), interp='nearest')
        c = Image.fromarray(fb_segs_argmax, mode='P')
        c.putpalette(img_palette)
        c.save(frame_name, "PNG", mode='P')


def mkdir(dl_path):
    if not os.path.exists(dl_path):
        print("path doesn't exist. trying to make %s" % dl_path)
        os.mkdir(dl_path)
    else:
        print('%s exists, cannot make directory' % dl_path)
        

def inf():
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    capsnet = CapsNet()
    with tf.Session(graph=capsnet.graph, config=gpu_config) as sess:
        capsnet.load(sess, config.save_file_inference % config.epoch_save)

        # loads in video
        video_name = '03deb7ad95'
        video, orig_dims = load_video(video_name + '.mp4')
        first_frame, img_palette = load_first_frame('00110.png')
        
        processed_first_frame = process_first_frame(first_frame)
        
        start_time = time.time()
        print('Starting Inference')
        
        generate_inference(sess, capsnet, video, processed_first_frame, orig_dims, video_name, img_palette)
        
        print('Finished Inference in %d(s)' % (time.time()-start_time))


if __name__ == "__main__":
    inf()
