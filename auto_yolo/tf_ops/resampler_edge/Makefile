# Requires tensorflow >= 1.6.0 (haven't tested on 1.5, definitely doesn't work on 1.4)
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Bit of an annoyance: tensorflow tries to import the cuda header as "cuda/include/cuda.h"
# but on computecanada, the directory structure is "cuda/<version>/include/cuda.h"
# and tensorflow is thus unable to find cuda.h. To get around this, we make a symbolic link called "cuda"
# in the home directory which points to the <version> directory. Create the link by doing:
# "cd ~ && ln -s $CUDA_HOME cuda". Then invoke the make command as: "make CUDA_INC_HACK=$HOME"
CUDA_INC_HACK="/usr/local/"

all: _resampler_edge.so

# The -DNDEBUG is to get around a bug in tensorflow that raises an error:
#     error constexpr function return is non-constant
#
# --expt-relaxed-constexpr addresses a separate issue and is required

resampler_edge_ops_gpu.cu.o: kernels/resampler_edge_ops_gpu.cu.cc kernels/resampler_edge_ops.h
	nvcc -std=c++11 -c -o resampler_edge_ops_gpu.cu.o kernels/resampler_edge_ops_gpu.cu.cc -Ikernels -I$(CUDA_INC_HACK) $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG

_resampler_edge.so: resampler_edge_ops_gpu.cu.o kernels/resampler_edge_ops.cc kernels/resampler_edge_ops.h
	g++ -std=c++11 -shared ops/resampler_edge_ops.cc resampler_edge_ops_gpu.cu.o kernels/resampler_edge_ops.cc -o _resampler_edge.so -fPIC -Ikernels -I$(CUDA_INC_HACK) $(TF_CFLAGS) $(TF_LFLAGS) -O2 -D GOOGLE_CUDA=1

clean:
	rm -rf resampler_edge_ops_gpu.cu.o _resampler_edge.so
