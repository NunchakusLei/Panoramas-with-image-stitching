# Panoramas-with-image-stitching
Course work for MM 805 - 3D Computer Vision (Winter 2018), Group Project @ University of Alberta, Canada

```
Authors:
  - Chenrui Lei
  - Anni Dai
```

# Stitching Pipeline
1. compositing
1. feature extraction
2. feature matching
3. panorama recognizing
4. mosaic images
5. image blending

# Dependencies
- OpenCV 3.2+
- Python 3
- PyQt5

# Modules
- **features.py**: provides a FeatureExtractor that could extract various features from an image.
- **matches.py**: provides a FeatureMatcher that could match two set of features.
- **stitching.py**: provides a Stitcher that stitch multiple images.
- **spherical.py**: provides an image warper that could warp image spherically.
- **cylindrical.py**: provides an image warper that could warp image cylindrically.
- **GUI.py**: provides an graphic user interface for this programe.

# Code Execution
Type in the following command on a terminal to execute the software with graphic user interface.

```bash
python3 GUI.py
```

# References
- Spherical warp: https://github.com/bluenight1994/CV5670/blob/b2c8e1af8133ac18504b18de3c80bb6d4bc695af/Assignment%203/Project3_AutoStitch/warp.py
- Cylindrical warp: https://github.com/TejasNaikk/Image-Alignment-and-Panoramas/blob/master/main.py
- seamlessClone: https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/
Test dataset: https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1
