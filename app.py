import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage
from model import DetectNet
from ultralytics import YOLO

# Loading pretrained model
yolo = YOLO("weights/best.pt")
save_dir = "detect.png"

# Instantiate model
model = DetectNet(yolo, save_name=save_dir)

def take_image():
    image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if image is not None:
        image = PILImage.open(image)
        image = image.convert("RGB")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image

def compute(image):
    result = model(image)
    if len(result) == 0:
        return "No crack found"
    elif len(result) == 1:
        area, score = result[0]
        return f"Crack predicted accuracy: {score:.2f} %\nThe area of crack is: {area:.2f} cm²"
    else:
        text = ""
        for i, out in enumerate(result):
            area, score = out
            text += f"Crack {i+1} predicted accuracy: {score:.2f} %\nThe area of crack {i+1} is: {area:.2f} cm²\n\n"
        return text

def main():
    st.title("Crack Detection App")
    image = take_image()
    if image is not None:
        result = compute(image)
        st.image(image, caption="Input Image", use_column_width=True)
        st.write(result)

if __name__ == "__main__":
    main()