import numpy as np
from astropy.io import fits
import os

from dps.datasets.base import (
    ImageDataset, ImageFeature, VariableShapeArrayFeature, ArrayFeature, StringFeature
)
from dps.utils import Param, atleast_nd, walk_images


class FITSDataset(ImageDataset):
    fits_file = Param()
    force_memmap = Param()
    image_shape = None

    """
    Params inherited from ImageDataset, listed here for clarity.

    postprocessing = Param("")
    tile_shape = Param(None)
    n_samples_per_image = Param(1)
    n_frames = Param(0)
    image_dataset_version = Param(1)

    """
    _artifact_names = ['depth']
    depth = None

    @property
    def features(self):
        if self._features is None:
            annotation_shape = (self.n_frames, -1, 7) if self.n_frames > 0 else (-1, 7)
            self._features = [
                ImageFeature("image", self.obs_shape, dtype=np.uint16),
                VariableShapeArrayFeature("annotations", annotation_shape),
                ArrayFeature("offset", (2,), dtype=np.int32),
                StringFeature("filename"),
            ]

        return self._features

    def _make(self):
        if os.path.isdir(self.fits_file):
            local_paths = walk_images(self.fits_file, 'fits')
            directory = self.fits_file
        else:
            local_paths = [os.path.basename(self.fits_file)]
            directory = os.path.dirname(self.fits_file)

        open_kwargs = dict(memmap=True) if self.force_memmap else {}

        for lp in local_paths:
            p = os.path.join(directory, lp)

            with fits.open(p, **open_kwargs) as hdul:
                image = atleast_nd(hdul[0].data, 3)

                if self.depth is None:
                    self.depth = image.shape[2]

                self._write_example(
                    image=image,
                    annotations=[],
                    filename=lp,
                )

        return dict(depth=self.depth)


if __name__ == "__main__":
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from astropy.utils.data import get_pkg_data_filename

    # image_file = get_pkg_data_filename('galactic_center/gc_2mass_k.fits')
    image_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')
    image_data = fits.getdata(image_file, ext=0)

    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()

    n = 32

    dset = FITSDataset(
        fits_file=image_file, postprocessing='tile_pad',
        n_samples_per_image=n, tile_shape=(200, 200),
        force_memmap=False,
    )
    print(dset.depth)

    sess = tf.Session()
    with sess.as_default():
        dset.visualize(n)
