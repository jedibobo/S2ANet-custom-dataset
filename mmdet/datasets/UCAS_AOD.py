import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from DOTA_devkit.ucas_aod_evaluation import voc_eval
from mmdet.core import norm_angle
from mmdet.core import rotated_box_to_poly_single
from mmdet.datasets.custom import CustomDataset
from .builder import DATASETS

import cv2


@DATASETS.register_module
class UCASAODDataset(CustomDataset):

    CLASSES = ("airplane", "car")

    def __init__(
        self,
        ann_file,
        pipeline,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
    ):
        super().__init__(
            ann_file,
            pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
        )
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = "{}.png".format(img_id)
            txtpath = osp.join(
                ("/").join(self.img_prefix.split("/")[:-2]),
                "Annotations",
                "{}.txt".format(img_id),
            )
            height, width = cv2.imread(osp.join(self.img_prefix, filename)).shape[:2]
            # print("height", height, "width", width)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height)
            )
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]["id"]
        txtpath = osp.join(
            ("/").join(self.img_prefix.split("/")[:-2]),
            "Annotations",
            "{}.txt".format(img_id),
        )
        with open(txtpath, "r") as f:
            lines = f.readlines()
            splitlines = [
                x.strip().replace("\t", " ").replace("  ", " ").split(" ")
                for x in lines
            ]
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for i, splitline in enumerate(splitlines):
                label = self.cat2label[splitline[0]]
                difficult = 0
                x1, y1, x2, y2, x3, y3, x4, y4 = [float(x) for x in splitline[1:9]]
                cx = (x1 + x2 + x3 + x4) / 4
                cy = (y1 + y2 + y3 + y4) / 4
                w = float(splitline[12])
                h = float(splitline[13])
                a = float(splitline[9])/180*np.pi
                new_w, new_h = max(w, h), min(w, h)
                a = a if w > h else a + np.pi / 2
                a = norm_angle(a)
                bbox = [cx, cy, new_w, new_h, a]
                # print(bbox)
                ignore = False

                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            if not bboxes:
                bboxes = np.zeros((0, 5))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes, ndmin=2)
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 5))
                labels_ignore = np.zeros((0,))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
                labels_ignore = np.array(labels_ignore)

            ann = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                bboxes_ignore=bboxes_ignore.astype(np.float32),
                labels_ignore=labels_ignore.astype(np.int64),
            )
            return ann

    def get_cat_ids(self, idx):

        cat_ids = []
        img_id = self.img_infos[idx]["id"]
        txtpath = osp.join(
            ("/").join(self.img_prefix.split("/")[:-2]),
            "Annotations",
            "{}.txt".format(img_id),
        )
        with open(txtpath, "r") as f:
            lines = f.readlines()
            splitlines = [
                x.strip().replace("\t", " ").replace("  ", " ").split(" ")
                for x in lines
            ]
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for i, splitline in enumerate(splitlines):
                label = self.cat2label[splitline[0]]
                cat_ids.append(label)
        return cat_ids

    def evaluate(self, results, work_dir=None, gt_dir=None, imagesetfile=None):
        results_path = osp.join(work_dir, "results_txt")
        mmcv.mkdir_or_exist(results_path)

        print("Saving results to {}".format(results_path))
        self.result_to_txt(results, results_path)

        detpath = osp.join(results_path, "{:s}.txt")
        annopath = osp.join(
            gt_dir, "{:s}.txt"
        )  # data/HRSC2016/Test/Annotations/{:s}.xml

        classaps = []
        map = 0

        for classname in self.CLASSES:
            rec, prec, ap = voc_eval(
                detpath,
                annopath,
                imagesetfile,
                classname,
                ovthresh=0.5,
                use_07_metric=True,
            )
            map = map + ap
            print(classname, ': ', ap)
            classaps.append(ap)

        map = map / len(self.CLASSES)
        print('map:', map)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)        
        # Saving results to disk
        with open(osp.join(work_dir, 'eval_results.txt'), 'w') as f:
            res_str = 'mAP:' + str(map) + '\n'
            res_str += 'classaps: ' + ' '.join([str(x) for x in classaps])
            f.write(res_str)
        return map

    def result_to_txt(self, results, results_path):
        img_names = [img_info['id'] for img_info in self.img_infos]
        assert len(results) == len(img_names), 'len(results) != len(img_names)'

        for classname in self.CLASSES:
            f_out = open(osp.join(results_path, classname + '.txt'), 'w')
            print(classname + '.txt')
            # per result represent one image
            for img_id, result in enumerate(results):
                for class_id, bboxes in enumerate(result):
                    if self.CLASSES[class_id] != classname:
                        continue
                    if bboxes.size !=0:
                        for bbox in bboxes:
                            score = bbox[5]
                            bbox = rotated_box_to_poly_single(bbox[:5])
                            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                                osp.splitext(img_names[img_id])[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                                bbox[5], bbox[6], bbox[7])
                            f_out.write(temp_txt)
            f_out.close()