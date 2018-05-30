# Requires tensorflow >= 1.6.0 (haven't tested on 1.5, definitely doesn't work on 1.4)
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

all: _render_sprites.so

render_sprites_ops_gpu.cu.o: kernels/render_sprites_ops_gpu.cu.cc kernels/render_sprites_ops.h
	nvcc -std=c++11 -c -o render_sprites_ops_gpu.cu.o kernels/render_sprites_ops_gpu.cu.cc -Ikernels -I$(CUDA_PARENT) $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

_render_sprites.so: render_sprites_ops_gpu.cu.o kernels/render_sprites_ops.cc kernels/render_sprites_ops.h
	g++ -std=c++11 -shared ops/render_sprites_ops.cc render_sprites_ops_gpu.cu.o kernels/render_sprites_ops.cc -o _render_sprites.so -fPIC -Ikernels $(TF_CFLAGS) $(TF_LFLAGS) -O2 -D GOOGLE_CUDA=1

clean:
	rm -rf render_sprites_ops_gpu.cu.o _render_sprites.so
