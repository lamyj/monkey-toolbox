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
    parser = argparse.ArgumentParser()
    parser.add_argument("t_image")
    parser.add_argument("mask_image")
    parser.add_argument("t_threshold", type=float)
    parser.add_argument(
        "p_threshold_or_cluster_size", type=float, 
        help="If <= 1, use as cluster-wise p-value threshold. "
            "If > 1 use as minimum number of voxels in cluster")
    parser.add_argument("clusters_image")
    # parser.add_argument("clusters_data")
    arguments = parser.parse_args()
    
    smoothness = subprocess.check_output([
        "smoothest", "-z", arguments.t_image, "-m", arguments.mask_image])
    smoothness = dict(x.split(" ", 1) for x in smoothness.decode().splitlines())
    
    with tempfile.TemporaryDirectory() as directory:
        min_cluster_size, clusters_data_pos = get_clusters(
            arguments.t_image, arguments.t_threshold, smoothness, 
            arguments.p_threshold_or_cluster_size, 
            "{}/pos.nii.gz".format(directory))
        clusters_image_pos = nibabel.load("{}/pos.nii.gz".format(directory))
        
        t_image = nibabel.load(arguments.t_image)
        nibabel.save(
            nibabel.Nifti1Image(-t_image.get_fdata(), t_image.affine),
            "{}/image.nii.gz".format(directory))
        
        _, clusters_data_neg = get_clusters(
            "{}/image.nii.gz".format(directory), arguments.t_threshold, 
            smoothness, arguments.p_threshold_or_cluster_size, 
            "{}/neg.nii.gz".format(directory))
        clusters_image_neg = nibabel.load("{}/neg.nii.gz".format(directory))
        
        clusters_image = nibabel.Nifti1Image(
            numpy.abs(clusters_image_pos.get_fdata() 
                - clusters_image_neg.get_fdata()),
            clusters_image_pos.affine)
    
    clusters_data = pandas.concat(
        [clusters_data_pos, clusters_data_neg], ignore_index=True)
    del clusters_data["Cluster Index"]
    clusters_data.sort_values("Voxels", ascending=False, inplace=True)
    clusters_data.reset_index(drop=True, inplace=True)
    
    # clusters_data.to_csv(arguments.clusters_data, index=False)
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
        if p_threshold_or_cluster_size <= 1:
            min_cluster_size, clusters = output.decode().split("\n", 1)
            min_cluster_size = int(
                re.search(r"(\d+)$", min_cluster_size).group(1))
        else:
            min_cluster_size = None
            clusters = output.decode()
        clusters_data = pandas.read_csv(io.StringIO(clusters), sep="\t")
    return min_cluster_size, clusters_data

if __name__ == "__main__":
    sys.exit(main())
