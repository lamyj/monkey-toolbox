import argparse
import sys

import nibabel
import numpy

def main():
    parser = argparse.ArgumentParser(
        "Create a left-right mirror of the source image")
    parser.add_argument("source")
    parser.add_argument("destination")
    arguments = parser.parse_args()

    source = nibabel.load(arguments.source)

    # Flip LR axis
    transform = numpy.diag([-1., 1., 1.])

    # Physical coordinates of image center
    center = source.affine @ [*numpy.divide(source.shape, 2.), 1.]
    center = center[:3]/center[3]

    # Original direction and offset
    direction = source.affine[:3, :3]
    origin = source.affine[:3, 3]

    # Build the new affine matrix with previous formula
    affine = numpy.full_like(source.affine, 0.)
    affine[:3, :3] = transform @ direction
    affine[:3, 3] = transform @ (origin - center) + center
    affine[3, 3] = 1

    destination = nibabel.Nifti1Image(source.dataobj, affine)
    nibabel.save(destination, arguments.destination)

if __name__ == "__main__":
    sys.exit(main())
