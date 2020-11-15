from os import path, listdir
from argparse import ArgumentParser
import json

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# from ELC import inference as elc
from hamsters_utils import PostProcessor
from PIL import Image


def makeImgList(dir):
    imgFileNames = listdir(dir)
    imgFileNames = [x for x in imgFileNames if x.endswith(('jpg', 'JPG', 'jpeg', 'JPEG'))]
    imgFileNames = [path.join(dir, x) for x in imgFileNames]

    return imgFileNames


def savePathFromImgPath(save_dir, imgPath):
    return path.join(save_dir, imgPath.split('/')[-1])


def inference(detectorsModel, detectorsPostProcessor, elcModel, elcPostProcessor, elcConfigs, imgList, args):
    for imgPath in imgList:
        result = inference_detector(detectorsModel, imgPath)
        # use just detectors
        if not elcModel:
            detectorsPostProcessor.saveResult(imgPath, result, show=False, out_file=savePathFromImgPath(args.save_dir, imgPath))
            detectorsPostProcessor.saveIitp(imgPath, result)

        # if use ELC module
        else:
            detectorsPostProcessor.saveResult(imgPath, result, show=False, out_file=savePathFromImgPath(args.save_dir, imgPath))
            croppedImgs = detectorsPostProcessor.cropBoxes(imgPath, result, out_file=savePathFromImgPath(args.save_dir, imgPath))
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
    parser.add_argument('--img_dir', help='Image files path',
                        default='/home/ubuntu/minseok/dataset/AI_Challenge/workFestival_padding_2/val2017/') # /home/ubuntu/minseok/dataset/AI_Challenge/workFestival_padding_2/train2017/
    parser.add_argument('--save_dir', help='path to save result file',
                        default='/home/ubuntu/minseok/mmdetection/results/dokyo_1')
    parser.add_argument('--detectors_config', help='Config file', 
                        default="/home/ubuntu/minseok/mmdetection/work_dirs/stuff/padding/semantic/detectors_htc_r50_1x_coco.py")
    parser.add_argument('--detectors_checkpoint', help='Checkpoint file',
                        default="/home/ubuntu/minseok/mmdetection/work_dirs/stuff/padding/semantic/epoch_17.pth")
    parser.add_argument('--score-thr', type=float, help='bbox score threshold', 
                        default=0.3)
    # ELC options
    parser.add_argument('--elc_checkpoint', help='post network checkpoint', 
                        )   # default="/home/ubuntu/namwon/new_test/superW-net1E/save_c27_1018/204_model_best.pth.tar"
    parser.add_argument('--use_att', default=True, action='store_true', help='use attention module')
    parser.add_argument('--nclass', default=27, type=int, help='number of class')
    parser.add_argument('--csvPath', default='/home/ubuntu/minseok/mmdetection/results/detectors_padding_2/t3_res_0026.csv', type=str, help='path to save csv result')
    # Device options
    parser.add_argument('--device', help='Device used for inference', 
                        default='cuda:2')
    args = parser.parse_args()

    # load image list
    imgList = makeImgList(args.img_dir)

    # build DetectoRS
    detectorsModel = init_detector(args.detectors_config, args.detectors_checkpoint, device=args.device)
    print('detectorsModel.CLASSES : ', detectorsModel.CLASSES)
    detectorsPostProcessor = PostProcessor(detectorsModel.CLASSES)

    # build ELC model
    elcModel = False
    elcPostProcessor = False
    elcConfigs = False
    if args.elc_checkpoint:
        elcModel, elcConfigs = elc.loadModel(args)
        elcPostProcessor = elc.ElcResultParser(args.csvPath)

    inference(detectorsModel, detectorsPostProcessor, elcModel, elcPostProcessor, elcConfigs, imgList, args)


if __name__ == '__main__':
    main()
