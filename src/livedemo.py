import os
import argparse

import torch
import cv2
import time

from test import GOTURN

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Man/img', type=str,
                    help='path to video frames')

def main(args):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    tester = GOTURN(args.data_directory,
                    args.model_weights,
                    device, True)
    # save initial frame with bounding box
    tester.model.eval()
    
    initBox = None
    try:
      init_rect = cv2.selectROI('Demo', tester.img[0][0], False, False)
      x, y, w, h = init_rect
      initBox = [x, y, w+x, h+y]
      print(initBox)
    except:
        exit()
    tester.set_init_box(initBox)

    count =0
    start=time.time()
    # loop through sequence images
    for i in range(tester.len):
        # get torch input tensor
        sample = tester[i]

        # predict box
        bbox = tester.get_rect(sample)
        #gt_bb = tester.gt[i]
        tester.prev_rect = bbox

        # save current image with predicted rectangle and gt box
        im = tester.img[i][1]
        sMatImageDraw = im.copy()
        cv2.rectangle(sMatImageDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        
        count+=1
        timeUsed = time.time()-start

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(sMatImageDraw,'{:0.2f}fps[#{}]'.format(count/timeUsed, i),(0,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.imshow('Demo', sMatImageDraw)
        key = cv2.waitKey(10)
        if key > 0:
            break
        # save(im, bb, gt_bb, i+2)

        # print stats
        # print('frame: %d, IoU = %f' % (
        #     i+2, axis_aligned_iou(gt_bb, bb)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
