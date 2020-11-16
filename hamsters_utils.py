  
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from random import randint

import cv2
import mmcv
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log
from mmcv.image import imread, imwrite

from mmdet.utils import get_root_logger


class PostProcessor():
    def __init__(self,
                classes,
                score_thr=0.3,
                thickness=1,
                font_scale=0.5,
                win_name='',
                wait_time=0):
        self.num_classes = len(classes)
        self.classes = classes
        self.score_thr = score_thr
        self.thickness = thickness
        self.font_scale = font_scale
        self.win_name = win_name
        self.wait_time = wait_time
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
                        (255, 0, 255), (255, 255, 255), (0, 0, 0), (255, 0, 128), (0, 191, 255),
                        (10, 255, 128), (191, 255, 0), (255, 191, 0), (255, 128, 10), (50, 152, 89)]
        self.makeColors()
        self.iitpID = 0
        self.iitpJson = {'annotations':[]}
        self.box_real_class = [0, 1, 1, 2, 3, 4, 5, 6, 5, 6]

    def makeColors(self):
        if len(self.colors) >= self.num_classes:
            return
        else:
            while len(self.colors) < self.num_classes:
                self.colors.append((randint(20, 230), randint(20, 230), randint(20, 230)))
            return

    def saveResult(self,
                    img,
                    result,
                    show=False,
                    out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)
        
        # draw bounding boxes
        return self.imshow_det_bboxes(img, bboxes, labels, show=show, out_file=out_file)

    def saveIitp(self, imgPath, result):
        _, bboxes, labels = self.extractInfo(imgPath, result, show=False, out_file=None)
        bboxes, labels = self.iitpProcess(bboxes, labels)
        self.annoMaker(imgPath, bboxes, labels)

    def labelChanger(self, labels):
        appliedLabels = []
        for i in labels:
            if i == 8:
                if 3 in labels or 4 in labels:
                    i = 3

            if i == 9:
                if 3 in labels:
                    i = 3
                elif 4 in labels:
                    i = 0
            i = self.box_real_class[i]
            appliedLabels.append(i)

        return labels

    def annoMaker(self, imgPath, bboxes, labels):
        anno = {}
        anno['id'] = self.iitpID
        self.iitpID+=1
        labels = self.labelChanger(labels)
        fileName = imgPath.split('/')[-1]
        anno['file_name'] = fileName
        anno['object'] = []
        for box, label in zip(bboxes, labels):
            anno['object'].append({
                'box': box,
                'label': 'c'+str(label)
                })
        self.iitpJson['annotations'].append(anno)

    def iitpProcess(self, bboxes, labels):
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        processedBoxes = []
        for box in bboxes:
            box = box.tolist()
            box.pop()
            box = list(map(int, box))
            processedBoxes.append(box)

        return processedBoxes, labels


    def cropBoxes(self, img, result, out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        img = np.ascontiguousarray(img)
        croppedImgs = []

        # path to save cropped image if save
        # splitPath = out_file.split('/')
        # fileName = splitPath.pop(-1).split('.')[0]

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):

            # !!!!!!!!!!! ~ Except class cap(8) or label(9) ~ !!!!!!!!!!!!
            if label != 8 and label != 9:
                
                bbox_int = bbox.astype(np.int32)
                heightRange = (bbox_int[1], bbox_int[3])
                widthRange = (bbox_int[0], bbox_int[2])

                dst = img.copy()
                dst = dst[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                croppedImgs.append(dst)

            # save cropped image            
            # out_file = splitPath.copy()
            # out_file.append(fileName+'_'+str(idx)+'.jpg')
            # out_file = '/'.join(out_file)
            # if out_file is not None:
            #     imwrite(dst, out_file)

        return croppedImgs

    def extractInfo(self,
                    img,
                    result,
                    show=False,
                    out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
                # print('check msrcnn : ', len(segm_result))
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            # print('check segm_result is not None')
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > self.score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        # if not (show or out_file):
        #     return img

        return img, bboxes, labels


    def imshow_det_bboxes(self,
                        img,
                        bboxes,
                        labels,
                        show=True,
                        out_file=None):

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        img = np.ascontiguousarray(img)
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, self.colors[label], thickness=self.thickness)
            label_text = self.classes[
                label] if self.classes is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - (label*2*randint(0, 1))),
                        cv2.FONT_HERSHEY_COMPLEX, self.font_scale, self.colors[label])

        if show:
            imshow(img, self.win_name, self.wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        return img
