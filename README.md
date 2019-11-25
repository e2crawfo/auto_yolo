This repository contains code for running experiments from the following paper:

Spatially Invariant Unsupervised Object Detection with Convolutional Neural Networks.  
Eric Crawford and Joelle Pineau.  
*AAAI (2019).*
```
@inproceedings{crawford2019spatially,  
  title={Spatiial Invariant Unsupervised Object Detection with Convolutional Neural Networks},  
  author={Crawford, Eric and Pineau, Joelle},  
  booktitle={Thirty-Third AAAI Conference on Artificial Intelligence},  
  year={2019}
}
```

This repository and the companion repository `dps` are both likely to undergo
further development in the future. In general, we will attempt to keep the
experiments from the paper runnable, but in case something breaks, one
can always check out the `aaai_2019` branches of both repositories to obtain
the code as it was for the paper. Also, branches for both repos
named `aaai_2019_v1` preserve most of the behaviour of the original experiments,
but with significant code improvements.

### Installation
1. [Install tensorflow](https://www.tensorflow.org/install/) with [GPU support](https://www.tensorflow.org/install/gpu).
   auto_yolo was developed with tensorflow 1.13.2 and CUDA 10.0; no guarantees that it will work
   with other versions, though it probably will.

2. `sh install.sh`

3. Install a version of `tensorflow_probability` that matches your version of tensorflow (0.6 works for tensorflow 1.13, increment by 0.1 for each 0.1 increment of tf version).

### Running Experiments
To train SPAIR on a scattered MNIST dataset:
```
cd auto_yolo/experiments/comparison
python yolo_air_run.py local
```
