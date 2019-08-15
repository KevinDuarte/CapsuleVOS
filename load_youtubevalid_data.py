import json
import os
from scipy.misc import imread, imresize
import numpy as np
from threading import Thread
import time
from PIL import Image
import sys
import matplotlib.pyplot as plt

data_loc = '/home/kevin/HD2TB/Datasets/YoutubeVOS2018/'
max_vids = 32


def get_split_names(tr_or_val):
    split_file = data_loc + '%s/meta.json' % tr_or_val

    all_files = []
    with open(split_file) as f:
        data = json.load(f)
        files = sorted(list(data['videos'].keys()))
        for file_name in files:
            all_files.append((file_name, data['videos'][file_name]['objects']))

    return all_files


# Loads in 8 frames
def load_video(file_name, tr_or_val='train', n_frames=8):
    video_dir = data_loc + ('%s_all_frames/JPEGImages/%s/' % (tr_or_val, file_name))

    # loads segmentations
    segment_dir = data_loc + ('%s/Annotations/%s/' % (tr_or_val, file_name))

    seg_frame_name = sorted(os.listdir(segment_dir))[0]
    seg_frame = np.array(Image.open(segment_dir + seg_frame_name))

    frame_names = sorted(os.listdir(video_dir))

    start_frame = frame_names.index(seg_frame_name[:-4] + '.jpg')

    # loads video
    frames = []
    for f in range(start_frame, start_frame+n_frames):
        try:
            frames.append(imread(video_dir + frame_names[f]))
        except:
            frames.append(frames[-1])

    video = np.stack(frames, axis=0)

    return video, seg_frame


def resize_video(video, segmentation, target_size=(120, 120)):
    frames, h, w, _ = video.shape

    t_h, t_w = target_size

    video_res = np.zeros((frames, t_h, t_w, 3), np.uint8)
    for frame in range(frames):
        video_res[frame] = imresize(video[frame], (t_h, t_w))

    segment_res = imresize(segmentation, (t_h, t_w), interp='nearest')

    return video_res/255., segment_res


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
        return 0, img.shape[0], 0, img.shape[1]

    return h0, h1, w0, w1


def perform_window_crop(gt_seg):
    h, w = gt_seg.shape

    # gets the bounds of the segmentation and the center of the object
    h0, h1, w0, w1 = get_bounds(gt_seg)
    obj_h, obj_w = h1 - h0, w1 - w0
    center_h, center_w = h0 + int(obj_h / 2), w0 + int(obj_w / 2)

    # defines the window size around the object
    if obj_h <= 0.2*h and obj_w <= 0.2*w:
        crop_dims = (0.25*h, 0.25*w)
    elif obj_h <= 0.4*h and obj_w <= 0.4*w:
        crop_dims = (0.5*h, 0.5*w)
    elif obj_h <= 0.65*h and obj_w <= 0.65*w:
        crop_dims = (0.75*h, 0.75*w)
    else:
        crop_dims = (h, w)

    y1 = max(0, center_h - crop_dims[0] / 2)
    x1 = max(0, center_w - crop_dims[1] / 2)
    if y1 + crop_dims[0] > h:
        y1 -= (y1+crop_dims[0]-h)
    if x1 + crop_dims[1] > w:
        x1 -= (x1+crop_dims[1]-w)

    return np.clip(np.array([y1/h, x1/w, crop_dims[0]/h, crop_dims[1]/w]), 0, 1)


def perform_window_crop2(gt_seg):
    h, w = gt_seg.shape

    # gets the bounds of the segmentation and the center of the object
    h0, h1, w0, w1 = get_bounds(gt_seg)
    obj_h, obj_w = h1 - h0, w1 - w0
    center_h, center_w = h0 + int(obj_h / 2), w0 + int(obj_w / 2)
    min_y, max_y, min_x, max_x = h0, h1, w0, w1

    leeway = 0.15
    leeway_h, leeway_w = leeway * h, leeway * w
    center_y1, center_x1 = h0 + int(obj_h / 2), w0 + int(obj_w / 2)
    h1, w1 = 2 * max(center_y1 - min_y, max_y - center_y1) + leeway_h, 2 * max(center_x1 - min_x,
                                                                               max_x - center_x1) + leeway_w
    h1, w1 = max(h1, h // 8), max(w1, w // 8)
    alpha1 = max(h1 / h, w1 / w)
    h1, w1 = alpha1 * h, alpha1 * w

    y1 = center_y1 - 0.5 * h1
    x1 = center_x1 - 0.5 * w1
    if y1 + h1 > h:
        y1 -= (y1 + h1 - h)
    if x1 + w1 > w:
        x1 -= (x1 + w1 - w)

    return np.clip(np.array([y1/h, x1/w, alpha1+0.01, 0]), 0, 1)


class YoutubeValidDataGen(object):
    def __init__(self, sec_to_wait=5, n_threads=10, crop_size=(256, 448), n_frames=8):
        self.train_files = get_split_names('valid')

        self.sec_to_wait = sec_to_wait

        self.crop_size = crop_size

        self.n_frames = n_frames

        self.data_queue = []

        self.thread_list = []
        for i in range(n_threads):
            load_thread = Thread(target=self.__load_and_process_data)
            load_thread.start()
            self.thread_list.append(load_thread)

        print('Waiting %d (s) to load data' % sec_to_wait)
        sys.stdout.flush()
        time.sleep(self.sec_to_wait)

    def __load_and_process_data(self):
        while self.train_files:
            while len(self.data_queue) >= max_vids:
                time.sleep(1)

            try:
                vid_name, fdict = self.train_files.pop()
            except:
                continue  # Thread issue

            video, segmentation = load_video(vid_name, tr_or_val='valid', n_frames=self.n_frames*2-1)

            video, segmentation = resize_video(video, segmentation, target_size=self.crop_size)

            # find objects in the first frame
            color, gt_seg = 0, 0
            for obj_id in sorted(fdict.keys()):
                color = int(obj_id)
                gt_seg = np.where(segmentation == color, 1, 0)
                if np.sum(gt_seg) > 0:
                    break
                else:
                    color = 0

            # no objects in the first frame
            if color == 0:
                print('%s has no foreground segmentation.' % vid_name)
                continue

            gt_crop = perform_window_crop2(gt_seg)

            gt_seg = np.expand_dims(gt_seg, axis=-1)

            seg_vid = [gt_seg]

            for i in range(self.n_frames*2-1 - 1):
                seg_vid.append(np.zeros_like(gt_seg))
            seg = np.stack(seg_vid, axis=0)

            self.data_queue.append((video, seg, gt_crop))

        print('Loading data thread finished')
        sys.stdout.flush()

    def get_batch(self, batch_size=5):
        while len(self.data_queue) < batch_size and self.train_files:
            print('Waiting on data')
            sys.stdout.flush()
            time.sleep(self.sec_to_wait)

        batch_size = min(batch_size, len(self.data_queue))
        batch_x, batch_seg, batch_crop = [], [], []
        for i in range(batch_size):
            vid, seg, crp = self.data_queue.pop(0)
            batch_x.append(vid)
            batch_seg.append(seg)
            batch_crop.append(crp)

        return batch_x, batch_seg, batch_crop

    def has_data(self):
        return self.data_queue != [] or self.train_files != []


# def main():
#     a = YoutubeValidDataGen(n_threads=1, crop_size=(256*2, 448*2))
#
#     while a.has_data():
#         v, s, cr1 = a.get_batch(1)
#         print(cr1)
#         cy, cx, alpha, _ = cr1[0]
#         cy, cx, h, w = int(cy*256*2), int(cx*448*2), int(alpha*256*2), int(alpha*448*2)
#
#         mask = np.ones_like(s[0][0, :, :, 0])*2
#         mask[cy:cy+h, cx:cx+w] = 0
#
#         plt.imshow(s[0][0, :, :, 0] + mask)
#         plt.show(plt)
#
#
#
#
# main()



