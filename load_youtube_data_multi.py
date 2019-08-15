import json
import os
from scipy.misc import imread, imresize
import numpy as np
import random
from threading import Thread
import time
from PIL import Image
import sys
#from scipy.signal import medfilt, medfilt2d
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter

data_loc = '/home/kevin/HD2TB/Datasets/YoutubeVOS2018/'
max_vids = 25


def get_split_names(tr_or_val):
    split_file = data_loc + '%s/meta.json' % tr_or_val

    all_files = []
    with open(split_file) as f:
        data = json.load(f)
        files = sorted(list(data['videos'].keys()))
        for file_name in files:
            fdict = data['videos'][file_name]['objects']
            frames = sorted(list(set([fr for x in fdict.keys() for fr in fdict[x]['frames']])))
            all_files.append((file_name, data['videos'][file_name]['objects'], frames))

    return all_files


def load_video(file_name, allowable_frames, tr_or_val='train', shuffle=True, n_frames=8, frame_skip=1):
    video_dir = data_loc + ('%s_all_frames/JPEGImages/%s/' % (tr_or_val, file_name))
    segment_dir = data_loc + ('%s_all_frames/Annotations/%s/' % (tr_or_val, file_name))

    frame_names = sorted(os.listdir(video_dir))
    seg_frame_names = sorted(os.listdir(segment_dir))
    frame_names = sorted([x for x in frame_names if x[:-4] + '.png' in seg_frame_names])

    start_ind = frame_names.index(allowable_frames[0] + '.jpg')

    if shuffle:
        start_frame = np.random.randint(start_ind, max(len(frame_names) - n_frames * frame_skip, start_ind + 1))
    else:
        start_frame = start_ind

    while start_frame > start_ind and frame_names[start_frame][:-4] not in allowable_frames:  # ensures the first frame is not interpolated
        start_frame -= 1

    # loads video
    frames = []
    for f in range(start_frame, start_frame + n_frames*frame_skip, frame_skip):
        try:
            frames.append(imread(video_dir + frame_names[f]))
            #frames.append(np.array(Image.open(video_dir + frame_names[f])))
        except:
            frames.append(frames[-1])

    video = np.stack(frames, axis=0)

    # loads segmentations
    frames = []
    for f in range(start_frame, start_frame + n_frames*frame_skip, frame_skip):
        try:
            frames.append(np.array(Image.open(segment_dir + seg_frame_names[f])))
        except:
            frames.append(frames[-1])

    segmentation = np.stack(frames, axis=0)

    return video, segmentation, frame_names[start_frame][:-4]


def resize_video(video, segmentation, target_size=(120, 120)):
    frames, h, w, _ = video.shape

    t_h, t_w = target_size

    video_res = np.zeros((frames, t_h, t_w, 3), np.uint8)
    segment_res = np.zeros((frames, t_h, t_w), np.uint8)
    for frame in range(frames):
        video_res[frame] = imresize(video[frame], (t_h, t_w))
        segment_res[frame] = imresize(segmentation[frame], (t_h, t_w), interp='nearest')

    return video_res/255., segment_res


def flip_clip(clip, segment_clip):
    flip_y = np.random.random_sample()
    #flip_x = np.random.random_sample()
    if flip_y >= 0.5:
        clip = np.flip(clip, axis=2)
        segment_clip = np.flip(segment_clip, axis=2)
    # if flip_x >= 0.5:
    #     clip = np.flip(clip, axis=1)
    #     segment_clip = np.flip(segment_clip, axis=1)

    return clip, segment_clip


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


def get_bounds2(img):
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


def get_bounds_frames(frames):
    y0, y1, x0, x1 = [], [], [], []

    y1_0, y2_0, x1_0, x2_0 = 0, 0, 0, 0

    for i in range(frames.shape[0]):
        h0, h1, w0, w1 = get_bounds2(frames[i])
        if i == 0 and h0 == -1:
            return -1, -1, -1, -1, -1, -1
        elif h0 == -1:
            continue
        else:
            y0.append(h0)
            y1.append(h1)
            x0.append(w0)
            x1.append(w1)

        if i == 0:
            y1_0, y2_0, x1_0, x2_0 = h0, h1, w0, w1

    center_y1, center_x1 = ((y2_0 + y1_0) / 2), ((x2_0 + x1_0) / 2)

    min_y, max_y, min_x, max_x = min(y0), max(y1), min(x0), max(x1)

    return center_y1, center_x1, min_y, max_y, min_x, max_x


def get_crops2(segmentations, n_frames, leeway=.15):
    _, h, w = segmentations.shape
    leeway_h, leeway_w = leeway * h, leeway * w

    y1_0, y2_0, x1_0, x2_0 = get_bounds(segmentations[0])
    y1_1, y2_1, x1_1, x2_1 = get_bounds(segmentations[n_frames - 1])
    y1_2, y2_2, x1_2, x2_2 = get_bounds(segmentations[n_frames*2 - 2])

    min_y = min(y1_0, y1_1)
    min_x = min(x1_0, x1_1)
    max_y = max(y2_0, y2_1)
    max_x = max(x2_0, x2_1)

    center_y1, center_x1 = ((y2_0 + y1_0)/2), ((x2_0 + x1_0)/2)
    h1, w1 = 2*max(center_y1 - min_y, max_y - center_y1) + leeway_h, 2*max(center_x1 - min_x, max_x - center_x1) + leeway_w
    h1, w1 = max(h1, h//8), max(w1, w//8)
    alpha1 = max(h1/h, w1/w)
    h1, w1 = alpha1*h, alpha1*w

    y1 = center_y1 - 0.5*h1
    x1 = center_x1 - 0.5*w1
    if y1 + h1 > h:
        y1 -= (y1+h1-h)
    if x1 + w1 > w:
        x1 -= (x1+w1-w)

    min_y = min(y1_2, y1_1)
    min_x = min(x1_2, x1_1)
    max_y = max(y2_2, y2_1)
    max_x = max(x2_2, x2_1)

    center_y2, center_x2 = ((y2_1 + y1_1) / 2), ((x2_1 + x1_1) / 2)
    h2, w2 = 2 * max(center_y2 - min_y, max_y - center_y2) + leeway_h, 2 * max(center_x2 - min_x, max_x - center_x2) + leeway_w
    h2, w2 = max(h2, h // 8), max(w2, w // 8)
    alpha2 = max(h2 / h, w2 / w)
    h2, w2 = alpha2 * h, alpha2 * w

    y2 = center_y2 - 0.5 * h2
    x2 = center_x2 - 0.5 * w2
    if y2 + h2 > h:
        y2 -= (y2+h2-h)
    if x2 + w2 > w:
        x2 -= (x2+w2-w)

    return np.clip(np.array([y1/h, x1/w, alpha1+0.01, 0]), 0, 1), np.clip(np.array([y2/h, x2/w, alpha2+0.01, 0]), 0, 1)


def get_crops_fin(segmentations, n_frames, leeway=0.15):
    n_in_frames, h, w = segmentations.shape
    leeway_h, leeway_w = leeway * h, leeway * w

    assert (n_in_frames - n_frames) % (n_frames-1) == 0

    crops = []
    i = 0

    while i < n_in_frames:
        frames = segmentations[i:i+n_frames]

        center_y1, center_x1, min_y, max_y, min_x, max_x = get_bounds_frames(frames)

        if center_y1 == -1:
            crops.append(crops[-1])
            i += 7
            continue

        h1, w1 = 2 * max(center_y1 - min_y, max_y - center_y1) + leeway_h, 2 * max(center_x1 - min_x, max_x - center_x1) + leeway_w
        h1, w1 = max(h1, h // 8), max(w1, w // 8)
        alpha1 = max(h1 / h, w1 / w)
        h1, w1 = alpha1 * h, alpha1 * w

        y1 = center_y1 - 0.5 * h1
        x1 = center_x1 - 0.5 * w1
        if y1 + h1 > h:
            y1 -= (y1 + h1 - h)
        if x1 + w1 > w:
            x1 -= (x1 + w1 - w)

        crops.append(np.clip(np.array([y1/h, x1/w, alpha1+0.01, 0]), 0, 1))
        i += 7

    return crops


class YoutubeTrainDataGen(object):
    def __init__(self, sec_to_wait=5, n_threads=10, crop_size=(256, 448), augment_data=True, n_frames=8,
                 rand_frame_skip=4, multi_objects=False):
        self.train_files = get_split_names('train')

        self.sec_to_wait = sec_to_wait

        self.augment = augment_data
        self.rand_frame_skip = rand_frame_skip

        self.crop_size = crop_size
        self.multi_objects = multi_objects

        self.n_frames = n_frames

        np.random.seed(None)
        random.shuffle(self.train_files)

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
                vid_name, fdict, allowable_frames = self.train_files.pop()
            except:
                continue  # Thread issue

            frame_skip = np.random.randint(self.rand_frame_skip) + 1

            video, segmentation, frame_name = load_video(vid_name, allowable_frames, tr_or_val='train', n_frames=self.n_frames*2-1, frame_skip=frame_skip)

            video_res, seg_res = resize_video(video, segmentation, self.crop_size)

            # find objects in the first frame
            allowable_colors = []
            for obj_id in fdict.keys():
                if frame_name in fdict[obj_id]['frames']:
                    allowable_colors.append(int(obj_id))

            # no objects in the first frame
            if len(allowable_colors) == 0:
                print(vid_name, 'has no colors - SHOULD BE IMPOSSIBLE - POSSIBLE BUG FOUND!')
                continue

            # selects the objects which will be chosen from clip
            colors_to_select = [x+1 for x in range(len(allowable_colors))]

            if self.multi_objects:
                n_colors_foreground = random.choice(colors_to_select)
            else:
                n_colors_foreground = 1

            random.shuffle(allowable_colors)
            selected_colors = allowable_colors[:n_colors_foreground]

            gt_seg = np.zeros_like(seg_res, dtype=np.float32)
            for color in selected_colors:
                gt_seg += np.where(seg_res == color, 1, 0)
            gt_seg = np.clip(gt_seg, 0, 1)

            #gt_seg = np.round(medfilt(gt_seg, (1, 3, 3)))
            for frame in range(gt_seg.shape[0]):
                if frame % 5 == 0:
                    continue
                gt_seg[frame] = np.round(median_filter(gt_seg[frame], 3))

            if np.sum(gt_seg[0]) == 0:
                print('No segmentation found. ERROR.')
                continue

            if self.augment:
                video_res, gt_seg = flip_clip(video_res, gt_seg)

            leeway = np.random.random_sample() * 0.1 + 0.15
            gt_crop1, gr_crop2 = get_crops2(gt_seg, self.n_frames, leeway=leeway)

            gt_seg = np.expand_dims(gt_seg, axis=-1)

            self.data_queue.append((video_res, gt_seg, gt_crop1, gr_crop2))
        print('Loading data thread finished')
        sys.stdout.flush()

    def get_batch(self, batch_size=5):
        while len(self.data_queue) < batch_size and self.train_files:
            print('Waiting on data. # Already Loaded = %s' % str(len(self.data_queue)))
            sys.stdout.flush()
            time.sleep(self.sec_to_wait)

        batch_size = min(batch_size, len(self.data_queue))
        batch_x, batch_seg, batch_crop1, batch_crop2 = [], [], [], []
        for i in range(batch_size):
            vid, seg, cr1, cr2 = self.data_queue.pop(0)
            batch_x.append(vid)
            batch_seg.append(seg)
            batch_crop1.append(cr1)
            batch_crop2.append(cr2)
            # print(cr1)
            # print(cr2)

        return batch_x, batch_seg, batch_crop1, batch_crop2

    def has_data(self):
        return self.data_queue != [] or self.train_files != []


#
# def main():
#     a = YoutubeTrainDataGen(n_threads=10, crop_size=(256*2, 448*2))
#     i = 0
#     while a.has_data():
#         v, s, cr1, cr2 = a.get_batch(1)
#         print(i)
#         i+=1
#         print(cr1)
#         cy, cx, h, w = cr1[0]
#         cy, cx, h, w = cy*256*2, cx*448*2, h*256*2, w*448*2
#
#         mask = np.ones_like(s[0][0, :, :, 0])*2
#         mask[np.clip(int(cy-0.5*h), 0, 256*2):np.clip(int(cy+0.5*h), 0, 256*2), np.clip(int(cx-0.5*w), 0, 448*2):np.clip(int(cx+0.5*w), 0, 448*2)] = 0
#
#         plt.imshow(s[0][0, :, :, 0] + mask)
#         plt.show(plt)
#         plt.imshow(s[0][7, :, :, 0] + mask)
#         plt.show(plt)
#
#         cy, cx, h, w = cr2[0]
#         cy, cx, h, w = cy * 256 * 2, cx * 448 * 2, h * 256 * 2, w * 448 * 2
#
#         mask = np.ones_like(s[0][0, :, :, 0]) * 2
#         mask[np.clip(int(cy - 0.5 * h), 0, 256 * 2):np.clip(int(cy + 0.5 * h), 0, 256 * 2),
#         np.clip(int(cx - 0.5 * w), 0, 448 * 2):np.clip(int(cx + 0.5 * w), 0, 448 * 2)] = 0
#
#         plt.imshow(s[0][7, :, :, 0] + mask)
#         plt.show(plt)
#         plt.imshow(s[0][7, :, :, -1] + mask)
#         plt.show(plt)
#
#
# main()
