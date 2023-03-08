import cv2
import numpy as np
import streamlit as st
from mmcls.apis import init_model
from mmcls.apis import inference_model_topk as inference_cls_model
from mmdet.registry import VISUALIZERS
# from mmcls.utils import register_all_modules as register_all_modules_cls
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules as register_all_modules_det
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

parent_folder = './'
topk = st.slider('Return top-k predictions', 1, 10, 3)

@st.cache_resource
def _init_model(model_option):
    if model_option == 'cls':
    # init model
        model = init_model(parent_folder + 'configs/resnet/resnet50_8xb32_in1k.py',
                       'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth')
        visualizer = None
    elif model_option == 'det':
        # register_all_modules_det()
        model = init_detector(parent_folder + 'configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py',
                          'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth', device='cpu')
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
    else:
        model = None
        visualizer = None
    return model, visualizer

@st.cache_data
def _get_image(my_upload=my_upload):
    if my_upload is not None:
        img_file = my_upload
    else:
        img_file = parent_folder + "images/zebra.jpg"
    return Image.open(img_file).convert('RGB')

# @st.cache_resource
def _inference_model(img, model, visualizer, model_option):
    img = np.array(img)
    if model_option == 'cls':
        return_results = inference_cls_model(model, img, 10)
        vis_img = img
    elif model_option == 'det':
        vis_img = img.copy()
        results = inference_detector(model, img)
        # import pdb
        # pdb.set_trace()
        b, h, w = results.pred_instances.masks.shape
        vis_img = cv2.resize(vis_img, (w, h))
        visualizer.add_datasample(
            name='result',
            image=vis_img,
            data_sample=results,
            draw_gt=False,
            show=False)
        vis_img = visualizer.get_image()
        cls_names = visualizer.dataset_meta['classes']
        return_results = {'scores': results.pred_instances.scores[:10].numpy(),
                          'bboxes': results.pred_instances.bboxes[:10].numpy(),
                          'labels': [cls_names[x.item()] for x in results.pred_instances.labels[:10]]
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


model, visualizer = _init_model(model_option)
img = _get_image(my_upload)
results, vis_img = _inference_model(img, model, visualizer, model_option)
plot_canvas(img, vis_img, results, model_option)
