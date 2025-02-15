This project was inspired by research done at [Georgia Tech][(https://faculty.cc.gatech.edu/~thad/p/030_30_AC/idc05_final.pdf] on ASL and computer vision. 
By creating a game to learn ASL, this project provides independence to ASL users so they can communicate without an interpreter and makes learning ASL accessible for everyone. 

This is the approach we used to create the game: 
1. Created dataset that used Mediapipe to normalize, extract, and draw hand landmarks on images

2. Trained Random Forest model over this dataset

3. Used OpenCV to connect to webcam and process the video input

4. Used MediaPipe to detect hand and finger gestures in real time for the game to simplify input 

5. Developed feedback metrics for interactive game element including comparing predicted letter to target letter

#### 1. Clone and Install

```bash
# Clone the repo
git clone https://github.com/prisharpatel/asl-detector.git

# Create and Train the Dataset
python3 create_dataset.py
python3 train_classifier.py

# Play
python3 inference_classifier.py
```

