class Header(Widget):
    pass


class OpenCamera(Camera):
    pass


class CrackContainer(ScreenManager):
    pass


class CrackContainer1(Screen):
    pass


class CrackContainer2(Screen):
    start_cam = ObjectProperty(None)
    close_cam = ObjectProperty(None)
    detect_img = ObjectProperty(None)
    report = ObjectProperty(None)

    def capture(self):
        print("Captured!")

        # Take out raw pixels
        camera = self.ids["camera"]
        raw = camera.texture.pixels
        size = camera.texture.size
        print(size)

        # Convert image to Tensor
        image = PILImage.frombuffer(mode="RGBA", size=size, data=raw)
        image = image.convert("RGB")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect and save
        result = model(image)

        # Display the detected image on the screen
        detected_image = self.ids["detect_img"]
        detected_image.source = save_dir

        # Print out result
        report = self.ids["report"]

        if len(result) == 0:
            report.text = "No crack found"
        elif len(result) == 1:
            area, score = result[0]
            report.text = (
                f"Crack predicted accuracy: "
                + "%.2f" % score
                + " %\nThe area of crack is: "
                + "%.2f" % area
                + " cm²"
            )
        else:
            text = ""
            for i, out in enumerate(result):
                area, score = out
                text += (
                    f"Crack {i+1} predicted accuracy: "
                    + "%.2f" % score
                    + f" %\nThe area of crack {i+1} is: "
                    + "%.2f" % area
                    + " cm²\n\n"
                )
            report.text = text

    def remove_img(self):
        detected_image = self.ids["detect_img"]
        detected_image.nocache = True
        detected_image.source = ""

    def remove_result(self):
        self.report.nocache = True
        self.report.text = ""


class CrackContainer3(Screen):
    pass


class CrackApp(App):
    def build(self):
        # Initialize container for whole app
        sm = ScreenManager(transition=NoTransition())
        screen1 = CrackContainer1(name="first")
        screen2 = CrackContainer2(name="second")

        sm.add_widget(screen1)
        sm.add_widget(screen2)

        return sm


if __name__ == "__main__":
    CrackApp().run()
#check