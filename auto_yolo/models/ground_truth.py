import tensorflow as tf

from auto_yolo.models import baseline


class GroundTruth_Network(baseline.Baseline_Network):
    cc_threshold = None

    def _build_program_generator(self):
        n_objects = self._tensors["n_annotations"] + 0

        _, top, bottom, left, right = tf.split(self._tensors["annotations"], 5, axis=2)
        height = bottom - top
        width = right - left

        _top = top / self.image_height
        _height = height / self.image_height

        _left = left / self.image_width
        _width = width / self.image_width

        self._tensors["normalized_box"] = tf.concat([_top, _left, _height, _width], axis=-1)
        self._tensors["obj"] = tf.to_float(tf.sequence_mask(n_objects)[:, :, None])
        self._tensors["n_objects"] = n_objects
        self._tensors["max_objects"] = tf.reduce_max(n_objects)
