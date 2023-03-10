import glob
import os.path

import cv2
import numpy as np
import streamlit as st
from mmcls.apis import inference_model, init_model
from mmdet.apis import inference_detector, init_detector
from PIL import Image

st.set_page_config(page_title="📷 A Folder Demo", page_icon="📷", layout='wide')
st.markdown("# 📷 A Folder Demo")
st.write(
    ":dog: Try uploading multi images to get the possible categories, objects."
)
st.sidebar.header("A Folder Demo")
my_upload = st.sidebar.file_uploader("Upload multi images",  type=["png", "jpg", "jpeg"], accept_multiple_files=True)

col1, col2 = st.columns(2)
parent_folder = './'
device = 'cpu'

@st.cache_resource
def _init_model_return_results(imgs):
    cls_model = init_model(
        parent_folder + 'cls_configs/resnet/resnet50_8xb32_in1k.py',
        parent_folder + 'pretrain/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        device=device
    )

    det_model = init_detector(
        parent_folder + 'det_configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py',
        parent_folder + 'pretrain/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth', device=device)

    imgs = [np.array(x) for x in imgs]
    results = {}

    for idx, img in enumerate(imgs):
        return_results = inference_model(cls_model, img, topk=3)
        results[idx] = set(np.array(return_results["pred_class"])[return_results["pred_score"] > 0.35])

        return_results = inference_detector(det_model, img)
        cls_names = det_model.CLASSES
        bbox_result, segm_result = return_results

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        score_thr = 0.25
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = bboxes[:, -1][:10]
        labels = np.array([cls_names[x.item()] for x in labels[:10]])
        results[idx] |= set(labels[scores > score_thr])

    class2idx = {}
    for k, v in results.items():
        for sub_v in v:
            class2idx[sub_v] = class2idx.get(sub_v, []) + [k]
    return results, class2idx


@st.cache_data
def _get_image(my_upload):
    if len(my_upload):
        img_files = my_upload
        if isinstance(img_files, str):
            img_files = [img_files]
        file_names = [os.path.basename(x.name).split('.')[0][:8] for x in img_files]
    else:
        img_files = glob.glob(parent_folder + "/images/*.jpg") + glob.glob(parent_folder + "/images/*.png")
        file_names = [os.path.basename(x).split('.')[0][:8] for x in img_files]

    return [Image.open(img_file).convert('RGB') for img_file in img_files], file_names


def plot_canvas(imgs, results, file_names, class2idx):
    col1.write("Original Images :camera:")
    col2.write("Filtered Images :wrench:")

    tabs = col1.tabs(file_names)
    for idx, tab in enumerate(tabs):
        tab.image(imgs[idx], width=400)

    all_classes = set()
    for x in results.values():
        all_classes |= x
    all_classes = list(all_classes)
    options = st.multiselect(
        'Select the classes:',
        all_classes)

    select_idx = set(range(len(file_names)))
    for idx, op in enumerate(options):
        select_idx &= set(class2idx[op])

    select_idx = np.array(list(select_idx))
    if len(select_idx):
        names = np.array(file_names)[select_idx].tolist()
        tabs = col2.tabs(names)
        for idx, tab in enumerate(tabs):
            tabs[idx].image(imgs[select_idx[idx]], width=400)
            tabs[idx].write(', '.join(results[select_idx[idx]]))


imgs, file_names = _get_image(my_upload)
results, class2idx = _init_model_return_results(imgs)
plot_canvas(imgs, results, file_names, class2idx)
