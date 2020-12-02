import argparse
import sys

import nibabel
import numpy

def main():
    parser = argparse.ArgumentParser(
        "Reorient data where the sample was placed feet-first-supine, but "
        "head-first-supine was specified on the scanner")
    parser.add_argument("source")
    parser.add_argument("destination")
    arguments = parser.parse_args()

    source = nibabel.load(arguments.source)

    # Mismatch between HFS and FFS means flipping the LR and rostro-caudal (mapped
    # to IS) axes around the image center

    # Since P = D⋅I + O (P: physical coordinates, I: index coordinates, D: direction 
    # matrix, O: origin, i.e. physical coordinate of I=(0,0,0)), given a transformed
    # T centered on C, its action on a point P is
    # T⋅(P - C) + C = T⋅(D⋅I + O) - T⋅C + C = (T⋅D)⋅I + T⋅(O-C) + C
    # The new direction matrix is hence T⋅D, and the new origin is T⋅(O-C) + C.

    # Transform flipping LR and IS (NIfTI is RAS)
    transform = numpy.diag([-1., 1., -1.])

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
