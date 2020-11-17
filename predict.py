from os import path, listdir
from argparse import ArgumentParser
import json
from tqdm import tqdm
import re

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# from ELC import inference as elc
from hamsters_utils import PostProcessor
from PIL import Image
import numpy as np


def makeImgList(dir):
    imgFileNames = listdir(dir)
    imgFileNames = [x for x in imgFileNames if x.endswith(('jpg', 'JPG', 'jpeg', 'JPEG'))]
    imgFileNames = [path.join(dir, x) for x in imgFileNames]

    return imgFileNames

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def savePathFromImgPath(save_dir, imgPath):
    return path.join(save_dir, imgPath.split('/')[-1])

def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag == 'color':
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag == 'grayscale':
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale" or "unchanged", '
                f'but got {flag}')
    return array

def makeImgPair(imgFileNames):
    for imgPath in imgFileNames:
        img = Image.open(imgPath)
        if img.width < img.height:
            img = img.transpose(Image.ROTATE_90)
        img = _pillow2array(img, flag='color', channel_order='bgr')
        yield imgPath, img

def inference(detectorsModel, detectorsPostProcessor, elcModel, elcPostProcessor, elcConfigs, imgList, SAVE_DIR, SCORE_CHECKER):
    if SCORE_CHECKER:
        f = open("cascade.csv", 'w')
        f.write("file_name,c1,c2,c3,c4,c5,c6,c7 \n")

    # for imgPath in imgList:
    for imgPath, img in tqdm(makeImgPair(imgList)):
        result = inference_detector(detectorsModel, img)
        # use just detectors
        if not elcModel:
            # detectorsPostProcessor.saveResult(img, result, show=False, out_file=savePathFromImgPath(SAVE_DIR, imgPath))
            detectorsPostProcessor.saveIitp(img, imgPath, result)

            if SCORE_CHECKER:
                # our score checker
                _, labels = detectorsPostProcessor.cropBoxes(img, result, out_file=None)
                output_class = [0] * 7
                f.write(imgPath.split("/")[-1])
                for label in labels:
                    output_class[label] = 1
                for i in output_class:
                    f.write("," + str(i))
                f.write("," + "\n")

        # if use ELC module
        else:
            detectorsPostProcessor.saveResult(imgPath, result, show=False, out_file=savePathFromImgPath(SAVE_DIR, imgPath))
            croppedImgs = detectorsPostProcessor.cropBoxes(imgPath, result, out_file=savePathFromImgPath(SAVE_DIR, imgPath))
            if len(croppedImgs) > 0:
                for idx, img in enumerate(croppedImgs):
                    pred_class, confidence = elc.inference(elcModel, elcConfigs, Image.fromarray(img))
                    elcPostProcessor(pred_class, confidence, imgPath.split('/')[-1])
            else:
                elcPostProcessor.iitp_csv[imgPath.split('/')[-1]] = [imgPath.split('/')[-1], 0, 0, 0, 0, 0, 0, 0]
    if elcModel:
        elcPostProcessor.saveCsv()

    with open('./t3_res_0022.json', 'w') as jsonFile:
        json.dump(detectorsPostProcessor.iitpJson, jsonFile)

def main():
    # DetectoRS options
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image files path')
    args = parser.parse_args()


    DETECTORS_CONFIG='./Chellange_detectors_cascade_rcnn_r50_1x_coco_WorkFestival_4_X3.py'
    DETECTORS_CHECKPOINT='./epoch.pth'
    SCORETHRESHOLD=0.5

    SCORE_CHECKER = False

    DEVICE='cuda:0'

    SAVE_DIR = '/msdet/testresults/'
    ELC_CHECKPOINT=None
    USE_ATT=True
    NCLASS=27
    CSVPATH='/home/ubuntu/minseok/mmdetection/results/detectors_padding_2/t3_res_0026.csv'
    ELC_ARGS = [ELC_CHECKPOINT, USE_ATT, NCLASS]
    

    # load image list
    imgList = sorted_aphanumeric(makeImgList(args.img_dir))

    # build DetectoRS
    detectorsModel = init_detector(DETECTORS_CONFIG, DETECTORS_CHECKPOINT, device=DEVICE)
    print('detectorsModel.CLASSES : ', detectorsModel.CLASSES)
    detectorsPostProcessor = PostProcessor(detectorsModel.CLASSES, score_thr=SCORETHRESHOLD)

    # build ELC model
    elcModel = False
    elcPostProcessor = False
    elcConfigs = False
    if ELC_CHECKPOINT:
        elcModel, elcConfigs = elc.loadModel(ELC_ARGS)
        elcPostProcessor = elc.ElcResultParser(CSVPATH)

    inference(detectorsModel, detectorsPostProcessor, elcModel, elcPostProcessor, elcConfigs, imgList, SAVE_DIR, SCORE_CHECKER)

if __name__ == '__main__':
    main()
