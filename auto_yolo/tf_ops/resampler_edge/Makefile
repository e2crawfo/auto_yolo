# Requires tensorflow >= 1.6.0 (haven't tested on 1.5, definitely doesn't work on 1.4)
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_HOME="/usr/local/cuda"
CUDA_PARENT=$(shell dirname $(CUDA_HOME))

all: _resampler_edge.so

resampler_edge_ops_gpu.cu.o: kernels/resampler_edge_ops_gpu.cu.cc kernels/resampler_edge_ops.h
	nvcc -std=c++11 -c -o resampler_edge_ops_gpu.cu.o kernels/resampler_edge_ops_gpu.cu.cc -Ikernels -I$(CUDA_PARENT) $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

_resampler_edge.so: resampler_edge_ops_gpu.cu.o kernels/resampler_edge_ops.cc kernels/resampler_edge_ops.h
	g++ -std=c++11 -shared ops/resampler_edge_ops.cc resampler_edge_ops_gpu.cu.o kernels/resampler_edge_ops.cc -o _resampler_edge.so -fPIC -Ikernels $(TF_CFLAGS) $(TF_LFLAGS) -O2 -D GOOGLE_CUDA=1

clean:
	rm -rf resampler_edge_ops_gpu.cu.o _resampler_edge.so
