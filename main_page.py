import streamlit as st
import os
os.system("mim install mmcv>=2.0.0rc4")
# https://huggingface.co/spaces/KyanChen/ai-photo-gallery

st.set_page_config(
    page_title="AI photo Gallery",
    page_icon="👋",
)

st.write("# AI Photo Gallery 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    AI Photo Gallery 👋！
    """
)