from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager, NoTransition
from kivy.uix.camera import Camera
from kivy.core.window import Window
Window.size = (430, 932)
import torch
import cv2
import numpy as np
from PIL import Image as PILImage
from kivy.uix.image import Image
from model import DetectNet
from ultralytics import YOLO
from kivy.uix.label import Label

import os

# Loading pretrained model
yolo = YOLO("weights/best.pt")
save_dir = "detect.png"

# Instantiate model
model = DetectNet(yolo, save_name=save_dir)


class Header(Widget):
    pass


class OpenCamera(Camera):
    pass

class CrackContainer1(Screen):
    start_cam = ObjectProperty(None)
    close_cam = ObjectProperty(None)
    detect_img = ObjectProperty(None)
    report = ObjectProperty(None)
    
    def take_image(self):
        # Take out raw pixels
        camera = self.ids["camera"]
        raw = camera.texture.pixels
        size = camera.texture.size

        # Convert image to Tensor
        image = PILImage.frombuffer(mode="RGBA", size=size, data=raw)
        image = image.convert("RGB")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return image

    def compute(self, image):
        # print("Captured!")

        # image = self.take_image()

        # Detect and save
        result = model(image)

        # Display the detected image on the screen
        detected_image = self.ids["detect_img"]
        detected_image.source = save_dir

        # Print out result
        report = self.ids["report"]

        if len(result) == 0:
            report.text = "No crack found"
        else:
            text = ""
            for i, out in enumerate(result):
                if len(out) >= 4:
                    area, score, length, width = out
                else:
                    print(f"Warning: Expected 4 values in output, but got {len(out)}")
                    continue  # Skip this iteration and move to the next one

                area /= 10000  # Convert area from cm² to m²

                text += (
                    f"Crack {i+1} predicted accuracy: {score:.2f} %\n"
                    + f"The area of crack {i+1} is: {area:.4f} m²\n\n"  # Area is already in square meters
                    + f"The length of crack {i+1} is: {length/100:.2f} m\n\n"  # Convert length from cm to m
                    + f"The width of crack {i+1} is: {width/100:.2f} m\n\n"  # Convert width from cm to m
                )
            report.text = text


    def remove_img(self):
        detected_image = self.ids["detect_img"]
        detected_image.nocache = True
        detected_image.source = ""

    def remove_result(self):
        self.report.nocache = True
        self.report.text = ""

    def choose_img(self):
        from plyer import filechooser
        path = filechooser.open_file()[0]
        image = PILImage.open(path)
        return image

class CrackApp(App):
    def build(self):
        sm = ScreenManager(transition=NoTransition())
        screen1 = CrackContainer1(name="first")
        sm.add_widget(screen1)
        return sm


if __name__ == "__main__":
    CrackApp().run()
#check
