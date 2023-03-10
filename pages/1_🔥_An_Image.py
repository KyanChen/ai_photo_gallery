import cv2
import mmcv
import numpy as np
import streamlit as st
from mmcls.apis import inference_model, init_model

from mmdet.apis import inference_detector, init_detector
import pandas as pd
from PIL import Image

st.set_page_config(page_title="🔥 An Image Demo", page_icon="🔥", layout='wide')
st.markdown("# 🔥 An Image Demo")
st.write(
    ":dog: Try uploading an image to get the possible categories, objects."
)
st.sidebar.header("An Image Demo")
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
col1, col2, col3 = st.columns(3)
model_option = st.radio(
    "What\'s your inference model",
    ('cls', 'det'))

parent_folder = '../'
topk = st.slider('Return top-k predictions', 1, 10, 3)
device = 'cpu'


@st.cache_resource
def _init_model(model_option):
    if model_option == 'cls':
        # init model
        model = init_model(
            parent_folder + 'cls_configs/resnet/resnet50_8xb32_in1k.py',
            parent_folder + 'pretrain/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            device=device
        )

    elif model_option == 'det':
        model = init_detector(
            parent_folder + 'det_configs/mask2former/mask2former_r101_lsj_8x2_50e_coco.py',
            parent_folder + 'pretrain/mask2former_r101_lsj_8x2_50e_coco_20220426_100250-c50b6fa6.pth', device=device)
    else:
        model = None
    return model

@st.cache_data
def _get_image(my_upload=my_upload):
    if my_upload is not None:
        img_file = my_upload
    else:
        img_file = parent_folder + "images/zebra.jpg"
    return Image.open(img_file).convert('RGB')

# @st.cache_resource
def _inference_model(img, model, model_option):
    img = np.array(img)
    if model_option == 'cls':
        return_results = inference_model(model, img, topk=10)
        vis_img = img
    elif model_option == 'det':
        results = inference_detector(model, img)
        if hasattr(model, 'module'):
            model = model.module
        score_thr = 0.3
        vis_img = model.show_result(
            img,
            results,
            score_thr=score_thr,
            bbox_color='coco',
            text_color=(200, 200, 200),
            mask_color='coco',
            thickness=2
        )
        bbox_result, segm_result = results
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # if segm_result is not None and len(labels) > 0:  # non empty
        #     segms = mmcv.concat_list(segm_result)
        #     if isinstance(segms[0], torch.Tensor):
        #         segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        #     else:
        #         segms = np.stack(segms, axis=0)
        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
        cls_names = model.CLASSES
        return_results = {'scores': bboxes[:, -1][:10],
                          'bboxes': bboxes[:, :4][:10],
                          'labels': [cls_names[x.item()] for x in labels[:10]]
                          }
    return return_results, vis_img


def plot_canvas(img, vis_img, results, model_option):
    col1.write("Original Image :camera:")
    col1.image(img)

    col2.write("Visualization:wrench:")
    col3.write("Metainfo:wrench:")
    if model_option == 'cls':
        col2.image(vis_img)
        df = pd.DataFrame({
            'category': results["pred_class"][:topk],
            'probability': [f"{x:.2f}" for x in results["pred_score"]][:topk]
            }, index=None)
        col3.dataframe(df)
    elif model_option == 'det':
        # vis_idx = st.slider('Show a prediction', 1, 10, 3, disabled=True)
        col2.image(vis_img)

        df = pd.DataFrame({
            'category': results["labels"][:topk],
            'probability': [f"{x:.2f}" for x in results["scores"]][:topk],
            'box': [list(map(lambda t: f"{t:.2f}", list(x))) for x in results["bboxes"][:topk]]
        }, index=None)
        col3.dataframe(df)


model = _init_model(model_option)
img = _get_image(my_upload)
results, vis_img = _inference_model(img, model, model_option)
plot_canvas(img, vis_img, results, model_option)
