import argparse
import io
import re
import subprocess
import sys
import tempfile

import nibabel
import numpy
import pandas

def main():
    parser = argparse.ArgumentParser(
        description="Create clusters from a VBA result, based on a t-statistic "
            "or z-score map and p-value or cluster size. A z-score map is "
            "required to use the p-value thresholding.")
    parser.add_argument("t_image", help="Path to the t or z map map")
    parser.add_argument(
        "mask_image", help="Path to a binary mask of the statistic map")
    parser.add_argument(
        "t_threshold", type=float, help="Lower threshold of the t or z map")
    parser.add_argument(
        "p_threshold_or_cluster_size", type=float, 
        help="If <= 1, use as cluster-wise p-value threshold; "
            "if > 1 use as minimum number of voxels in cluster.")
    parser.add_argument(
        "clusters_image", help="Path to the resulting cluster image")
    arguments = parser.parse_args()
    
    smoothness = subprocess.check_output([
        "smoothest", "-z", arguments.t_image, "-m", arguments.mask_image])
    smoothness = dict(x.split(" ", 1) for x in smoothness.decode().splitlines())
    
    with tempfile.TemporaryDirectory() as directory:
        get_clusters(
            arguments.t_image, arguments.t_threshold, smoothness, 
            arguments.p_threshold_or_cluster_size, 
            "{}/pos.nii.gz".format(directory))
        clusters_image_pos = nibabel.load("{}/pos.nii.gz".format(directory))
        
        t_image = nibabel.load(arguments.t_image)
        nibabel.save(
            nibabel.Nifti1Image(-t_image.get_fdata(), t_image.affine),
            "{}/image.nii.gz".format(directory))
        
        get_clusters(
            "{}/image.nii.gz".format(directory), arguments.t_threshold, 
            smoothness, arguments.p_threshold_or_cluster_size, 
            "{}/neg.nii.gz".format(directory))
        clusters_image_neg = nibabel.load("{}/neg.nii.gz".format(directory))
        
        clusters_image = nibabel.Nifti1Image(
            numpy.abs(clusters_image_pos.get_fdata() 
                - clusters_image_neg.get_fdata()),
            clusters_image_pos.affine)
    
    nibabel.save(clusters_image, arguments.clusters_image)

def get_clusters(
        source, t_threshold, smoothness, p_threshold_or_cluster_size, output):
    with tempfile.TemporaryDirectory() as directory:
        command = [
            "cluster",
            "--in={}".format(source), "--thresh={}".format(t_threshold), 
            "--othresh={}".format(output)]
        if p_threshold_or_cluster_size <= 1:
            command += [
                "--volume={}".format(smoothness["VOLUME"]), 
                "--dlh={}".format(smoothness["DLH"]), 
                "--resels={}".format(smoothness["RESELS"]), "--minclustersize", 
                "--pthresh={}".format(p_threshold_or_cluster_size)]
        else:
            command += ["--minextent={}".format(
                int(numpy.round(p_threshold_or_cluster_size)))]
        output = subprocess.check_output(command)

if __name__ == "__main__":
    sys.exit(main())
