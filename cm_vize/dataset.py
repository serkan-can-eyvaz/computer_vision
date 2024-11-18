from roboflow import Roboflow
rf = Roboflow(api_key="xa8fHgkikIT9Y6nlvVTD")
project = rf.workspace("kays-rsdol").project("cv_vize2")
version = project.version(1)
dataset = version.download("yolov8")