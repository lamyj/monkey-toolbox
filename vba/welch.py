import argparse
import glob
import os
import sys

import nibabel
import numpy
import pandas
import scipy.stats

def main():
    parser = argparse.ArgumentParser(
        description="Voxel-wise Welch's t-test, computing the raw t-statistic, "
        "the p-value, and the z-score maps")
    parser.add_argument(
        "table", help="Path to the data table with image paths (optionally "
            "including glob patterns) and grouping information in CSV or "
            "XLS/XLSX format")
    parser.add_argument(
        "group", help="Name of the grouping column of the table")
    parser.add_argument(
        "t_path", metavar="t.nii.gz", help="Path to the output t-statistic map")
    parser.add_argument(
        "p_path", nargs="?",
        metavar="p.nii.gz", help="Path to the output p-value map")
    parser.add_argument(
        "z_path", nargs="?",
        metavar="z.nii.gz", help="Path to the output z-score map")
    parser.add_argument(
        "--root", 
        help="Path to the directory containing the images, if they are listed "
            "in the table as relative paths",
        default=os.path.abspath(os.getcwd()))
    parser.add_argument(
        "--mask", 
        help="Path to mask image (only non-zero voxels will be processed)")
    
    arguments = parser.parse_args()
    
    try:
        arguments.table = pandas.read_excel(arguments.table)
    except:
        arguments.table = pandas.read_csv(arguments.table)
    
    if "Image" not in arguments.table.columns:
        parser.error("Mandatory column \"Image\" is missing from data table")
    
    if arguments.group not in arguments.table.columns:
        parser.error(
            "Mandatory column \"{}\" is missing from data table".format(
                arguments.group))
    
    for _, row in arguments.table.iterrows():
        if not os.path.isabs(row["Image"]):
            row["Image"] = os.path.join(arguments.root, row["Image"])
        matches = glob.glob(row["Image"])
        if len(matches) == 0:
            parser.error("No image matching {}".format(row["Image"]))
        elif len(matches) > 1:
            parser.error("Multiple images matching {}: {}".format(
                row["Image"], matches))
        row["Image"] = matches[0]
    
    arguments = vars(arguments)
    del arguments["root"]
    welch(**arguments)

def welch(table, group, t_path, p_path, z_path, mask):
    # Load and transpose the image data so that the resulting 4D array is a
    # vector image.
    images = [nibabel.load(x) for x in table["Image"]]
    image_data = numpy.ascontiguousarray(
            [x.get_fdata() for x in images]
        ).transpose(1,2,3,0)
    
    if mask is not None:
        mask = nibabel.load(mask)
        # numpy mask is 1 for invalid values
        mask = numpy.asarray(mask.dataobj)==0
    
        # Get the unmasked voxels. NOTE: the return value of compressed must be
        # reshaped to a 2D array.
        image_data = numpy.ma.array(
            image_data, 
            mask=numpy.broadcast_to(mask[...,None], image_data.shape))
        flat_data = image_data.compressed().reshape(-1, len(images))
    else:
        flat_data = image_data.reshape(-1, len(images))

    # Get the group members
    groups = table[group].astype("category").cat.codes
    groups_count = len(groups.unique())
    if groups_count == 1:
        raise Exception("Expected two groups, only one was specified")
    elif groups_count > 2:
        raise Exception(
            "Expected two groups, {} were specified".format(groups_count))
    groups = (groups == 0).array, (groups == 1).array
    groups = [flat_data[..., g] for g in groups]
    
    arrays = {}
    arrays["t"], arrays["p"] = scipy.stats.ttest_ind(
        groups[0], groups[1], axis=1, equal_var=False)
    # Compute the associated z-score from the p-value.
    # ppf(q) is the quantile corresponding to the CDF of the lower tail (i.e.
    # <0 for low values of q, >0 for high values of q). The Z interval is the 
    # given by [ppf(0.5 * p), -ppf(0.5*p)] and the signed Z-value will then have
    # the sign of the t-statistic. Since ppf(0.5*p) is always <0, we get:
    arrays["z"] = (
        -scipy.stats.norm.ppf(0.5*arrays["p"]) 
        * numpy.sign(arrays["t"]))
    
    # Recreate arrays from masked/flat data
    for name, flat_array in arrays.items():
        if mask is not None:
            array = numpy.zeros_like(image_data[...,0])
            array[~array.mask] = flat_array
            array = numpy.asarray(array)
        else:
            array = flat_array.reshape(image_data.shape[:-1])
        arrays[name] = array
    
    # Write the NIfTI images
    for name, array in arrays.items():
        path = locals()["{}_path".format(name)]
        if path is None:
            continue
        nibabel.save(nibabel.Nifti1Image(array, images[0].affine), path)

if __name__ == "__main__":
    sys.exit(main())
