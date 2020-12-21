"""
IITP 01: evaluation.py
"""
import os
import sys
import json
import pandas as pd
from operator import itemgetter

MINOVERLAP = 0.75


def voc_ap(rec, prec):

    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def compute_AP(tp, fp, gt_len):

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_len
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap

def cal_IoU(ground_truth_data, dr_data):

    if len(ground_truth_data) == 0:
        if len(dr_data) == 0:
            return [], []
        else:
            return [0]*len(dr_data), [1]*len(dr_data)
    else:
        if len(dr_data) == 0:
            return [], []

    tp, fp = [0]*len(dr_data), [0]*len(dr_data)
    used = [0]*len(ground_truth_data)

    for idx, bb in enumerate(dr_data):
        ovmax = -1
        gt_match = -1

        for jdx, obj in enumerate(ground_truth_data):
            bbgt = obj[:4]
            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw > 0 and ih > 0:
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                ov = iw * ih / ua

                if ov > ovmax:
                    ovmax = ov
                    gt_match = jdx

        min_overlap = MINOVERLAP
        if ovmax >= min_overlap:
            if gt_match != -1 and used[gt_match] == 0:
                tp[idx] = 1
                used[gt_match] = 1
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    return tp, fp


def track1_evaluation(gt, pred):

    # load ground truths
    with open(gt, 'r') as json_file:
        gt_data = json.load(json_file)

    ground_truths = dict()
    gt_len = 0
    for img in gt_data['annotations']:
        img_id = img['file_name']
        if img_id not in ground_truths:
            ground_truths[img_id] = []
        for obj in img['box']:
            ground_truths[img_id].append(obj['position'])

        gt_len += len(img['box'])
    gt_img_li = list(ground_truths.keys())

    # predictions
    with open(pred, 'r') as json_file:
        pred_data = json.load(json_file)

    img_annots = pred_data['annotations']

    if len(img_annots) == 0: # exception handling
        print('Empty annotations')
        return 0

    predictions = []
    for img in img_annots:
        img_fn = img['file_name']
        img_index = len(predictions)

        img_dt = []

        if len(img['box']) == 0: # exception handling
            continue
        
        # sort bboxes by confidence score
        sort_boxes = sorted(img['box'], key=itemgetter('confidence_score'), reverse=True)
        
        for obj in sort_boxes:
            try:
                # parse detections
                bbox = {}
                bbox['file_name'] = img_fn
                if len(obj['position']) != 4: # exception handling
                    print("Position value error")
                    raise ValueError

                bbox['position'] = obj['position']
                img_dt.append(obj['position'])
                bbox['confidence_score'] = float(obj['confidence_score'])
                bbox['tp'] = 0
                bbox['fp'] = 0
                predictions.append(bbox)

            except Exception as e:
                print(f'Parsing Error {e} {img_fn}')
                raise KeyError

        try:
            img_gt = ground_truths[img_fn]
        except Exception as e: # exception handling
            print(f'Filename not found {e} {img_fn}')
            raise ValueError

        # get TP/FP
        tp, fp = cal_IoU(img_gt, img_dt)

        for jdx in range(len(img_dt)):
            predictions[img_index + jdx]['tp'] = tp[jdx]
            predictions[img_index + jdx]['fp'] = fp[jdx]

    if len(predictions) == 0: # exception handling
        print('Empty list of predictions')
        return 0
        
    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values(by=['confidence_score'], ascending=False)

    true_positives = list(df_pred['tp'])
    false_positives = list(df_pred['fp'])

    ap = compute_AP(true_positives, false_positives, gt_len)

    return ap


if __name__ == '__main__':

    gt = sys.argv[1]
    pred = sys.argv[2]

    score = track1_evaluation(gt, pred)
    print('score:', score)
