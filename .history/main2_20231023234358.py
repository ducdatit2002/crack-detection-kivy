from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder


Builder.load_string("""
<MyWidget>:
    id: my_widget
    FileChooserListView:
        id: filechooser
        on_selection: my_widget.selected(filechooser.selection)
    Image:
        id: image
        source: ""
""")


class MyWidget(BoxLayout):

    def selected(self,filename):
        self.ids.image.source = filename[0]


class MyApp(App):
    def build(self):
        return MyWidget()


if name == 'main2':
    MyApp().run()