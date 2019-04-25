This repository contains code for running experiments from the following paper:

Spatially Invariant Unsupervised Object Detection with Convolutional Neural Networks.  
Eric Crawford and Joelle Pineau.  
*AAAI (2019).*

This repository and the companion repository `dps` are both likely to undergo
further development in the future. In general, we will attempt to keep the
experiments from the paper runnable, but in case something breaks, one
can always check out the `aaai_2019` branches of both repositories to obtain
the code as it was for the paper. Also, branches for both repos
named `aaai_2019_v2` preserve most of the behaviour of the original experiments,
but with significant code improvements.

### Installation
1. [Install TensorFlow](https://www.tensorflow.org/install/) with [GPU support](https://www.tensorflow.org/install/gpu). auto_yolo should work with any version of TensorFlow > 1.4, but has not been tested extensively with any version other than 1.8.

2. Clone `dps`, optionally switch to `aaai_2019` or `aaai_2019_v2` branches, and install:
    ```
    git clone https://github.com/e2crawfo/dps.git
    cd dps
    (optional: git checkout aaai_2019 or git checkout aaai_2019_v2)
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```

3. Clone `auto_yolo`, optionally switch to `aaai_2019` or `aaai_2019_v2` branches, and install:
    ```
    git clone https://github.com/e2crawfo/auto_yolo.git
    cd auto_yolo
    (optional: git checkout aaai_2019 or git checkout aaai_2019_v2)
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```

4. Compile custom TensorFlow ops `resampler_edge` and `render_sprites`.
    ```
    cd auto_yolo/auto_yolo/tf_ops/resampler_edge && make
    cd ../render_sprites && make
    cd ../../../../
    ```

5. Setup scratch directory and download emnist data.
    ```
    cd dps/scripts
    python download.py emnist --shape=14,14
    ```
    The last line should first ask you for a scratch directory on your computer. It will then download emnist data into that directory, and reshape it to (14, 14) (this process can take a while).


### Running Experiments
To train SPAIR on a scattered MNIST dataset:
```
cd auto_yolo/experiments/comparison
python yolo_air_run.py
```
