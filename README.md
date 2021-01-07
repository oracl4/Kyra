# Kyra License Plate and Face Recognition

This is the Kyra License Plate and Face Recognition System for Indonesian License Plate.

### Installation

KyraWeb requires [OpenALPR](https://github.com/openalpr/openalpr) and [face_recognition](https://github.com/ageitgey/face_recognition) Library to run.

Clone the project and start the main.py

```sh
$ git clone https://github.com/oracl4/kyra.git
$ cd kyra
$ python main.py
```

Run the flask server for Web GUI
```sh
$ python main_web.py
```

> Configure the RTSP URL to match your IP Camera.

> Place the face dataset inside dataset/Faces/ folder and call the fr_encode function in the FaceRecog API to encode the faces and train the classifier.

> To run testing place the testing dataset in dataset/Testing/ folder and run each script in Research folder to test each function.
