Code for running experiments from the following paper

Spatially Invariant Unsupervised Object Detection with Convolutional Neural Networks.
Eric Crawford and Joelle Pineau.
AAAI (2019).

Installation
------------
1. Install TensorFlow with CUDA support (auto_yolo was developed with version 1.8) using method of choice. (e.g. pip install tensorflow-gpu==1.8)

2. Clone dps, switch to auto_yolo branch, and install:
  git clone https://github.com/e2crawfo/dps.git
  cd dps
  git checkout auto_yolo
  pip install -r requirements.txt
  pip install -e .
  cd ..

3. Clone auto_yolo and install:
  git clone https://github.com/e2crawfo/auto_yolo.git
  cd auto_yolo
  pip install -r requirements.txt
  pip install -e .
  cd ..

4. Compile custom tensorflow ops resampler_edge and render_sprites.
  cd auto_yolo/auto_yolo/tf_ops/resampler_edge && make
  cd ../render_sprites && make
  cd ../../../../

5. Setup scratch directory and download emnist data.
  cd dps/scripts
  python download.py emnist --shape=14,14

  The last line should first ask you for a scratch directory on your computer.
  It will then download emnist data into that directory, and reshape it to (14, 14)
  (this process can take a while).

Running Experiments
-------------------
cd auto_yolo/experiments/comparison
python yolo_air_run.py
