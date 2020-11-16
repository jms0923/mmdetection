from os import path, listdir
from argparse import ArgumentParser
import json

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


def savePathFromImgPath(save_dir, imgPath):
    return path.join(save_dir, imgPath.split('/')[-1])

def makeImgPair(imgFileNames):
    for imgPath in imgFileNames:
        img = Image.open(imgPath)
        print('type img : ', type(img))
        if img.width < img.height:
            img = img.rotate(90)
        img = np.array(img)
        print('type img : ', type(img))
        print()
        yield imgPath, img

def inference(detectorsModel, detectorsPostProcessor, elcModel, elcPostProcessor, elcConfigs, imgList, SAVE_DIR):
    # for imgPath in imgList:
    for imgPath, img in makeImgPair(imgList):
    
        result = inference_detector(detectorsModel, img)
        # use just detectors
        if not elcModel:
            # detectorsPostProcessor.saveResult(imgPath, result, show=False, out_file=savePathFromImgPath(SAVE_DIR, imgPath))
            detectorsPostProcessor.saveIitp(imgPath, result)

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
    parser.add_argument('img_dir', help='Image files path') # , default='/home/ubuntu/minseok/dataset/AI_Challenge/workFestival_padding_2/val2017/'
    args = parser.parse_args()


    SAVE_DIR = '/home/ubuntu/minseok/mmdetection/results/dokyo_1'

    DETECTORS_CONFIG='/aichallenge/detectors_htc_r50_1x_coco.py'
    DETECTORS_CHECKPOINT='/aichallenge/epoch.pth'
    SCORETHRESHOLD=0.5
    
    ELC_CHECKPOINT=None
    USE_ATT=True
    NCLASS=27
    CSVPATH='/home/ubuntu/minseok/mmdetection/results/detectors_padding_2/t3_res_0026.csv'
    ELC_ARGS = [ELC_CHECKPOINT, USE_ATT, NCLASS]

    DEVICE='cuda:0'
    

    # load image list
    imgList = makeImgList(args.img_dir)

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

    inference(detectorsModel, detectorsPostProcessor, elcModel, elcPostProcessor, elcConfigs, imgList, SAVE_DIR)


if __name__ == '__main__':
    main()
