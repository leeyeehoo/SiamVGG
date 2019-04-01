import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import collections
import math
from utils import gen_xz,get_zbox,get_xbox,convert_array_to_rec,convert_bbox_format,bbox_iou

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

def compute_iou(anchors, box):
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def load_data(pair_infos,discrim,train = True):
    
    
    
    
    
    if random.random>0.0:

        img_path1 = pair_infos[0][0]
        img_path2 = pair_infos[1][0]
    
        bs1 = pair_infos[0][1]#xmin xmax ymin ymax
        bs2 = pair_infos[1][1]
        
        
        
        gt1 = Rectangle(bs1[0],bs1[2],bs1[1]-bs1[0],bs1[3]-bs1[2])
        gt2 = Rectangle(bs2[0],bs2[2],bs2[1]-bs2[0],bs2[3]-bs2[2])
    
        gt1 = convert_bbox_format(gt1, to = 'center-based')
        gt2 = convert_bbox_format(gt2, to = 'center-based')
    
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
    
        zbox1 = get_zbox(gt1,0.25)
        zbox2 = get_zbox(gt2,0.25)
        
        
    else:
        cats = coco.loadCats(coco.getCatIds())
        nms=[cat['name'] for cat in cats]
        # catIds = coco.getCatIds(nms[0])
        # catIds = coco.getCatIds('bicycle')
        catIds = coco.getCatIds(nms[np.random.randint(0,len(nms))]);
        imgIds = coco.getImgIds(catIds=catIds );
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if i['category_id'] == catIds:
                break
        gt_org = convert_array_to_rec(ann['bbox'])
        gt = convert_bbox_format(gt_org, to = 'center-based')
        zbox = get_zbox(gt,0.25)
        xbox = get_xbox(zbox)
        img1 = Image.open(dataDir+ dataType+ '/%012d.jpg'%img['id'])

        imgIds = coco.getImgIds(catIds=catIds );
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if i['category_id'] == catIds:
                break
        gt_org = convert_array_to_rec(ann['bbox'])
        gt = convert_bbox_format(gt_org, to = 'center-based')
        zbox = get_zbox(gt,0.25)
        xbox = get_xbox(zbox)
        img2 = Image.open(dataDir+ dataType+ '/%012d.jpg'%img['id'])
        gt[:,:,:] = -1
    
    scales_w = 1.04**(random.random()*6-3)
    scales_h = 1.04**(random.random()*6-3)
    
    
    
    #scales_w = 2
    #scales_h = 2
    
    zbox2_scaled = Rectangle(zbox2.x,zbox2.y,zbox2.width*scales_w,zbox2.height*scales_h)
    
    #dx = 1-2*random.random()
    
    #dy = 1-2*random.random()
    
    #dx = 1
    
    #dy = 1
    
    #dx = dx/2
    #dy = dy/2
    
    dx = 0
    dy = 0
    
    xbox2 = get_xbox(zbox2_scaled,dx,dy)# we assume second is the search region
    #print zbox2,zbox2_scaled
    
    
    
    z = gen_xz(img1,zbox1,to='z')
    x = gen_xz( img2, xbox2,to = 'x')

    info = [dx,dy,gt2.width/scales_w/zbox2.width, gt2.height/scales_h/zbox2.height]
    

    
    #gt_box = np.array([-info[0]*64,-info[1]*64,info[2]*128,info[3]*128])
    gt_box = np.array([np.log(info[2]*2),np.log(info[3]*2)])
    
    gt = np.zeros((1,17,17))
    gt[:,:,:] = -1
    gt[0,8,8] = 1.
    
    gt[0,7:10,7:10] = 1.
    gt[0,8:9,6:11] = 1.
    gt[0,6:11,8:9] = 1.
    
    gt = np.zeros((1,28,28))
    
    gt[0,13:15,13:15] = 1.
    
    return z,x,gt,gt_box
    
    #return z,x,segz,gt