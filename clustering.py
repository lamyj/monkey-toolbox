import os
import subprocess
import tempfile

import nibabel
import numpy

def by_p_value(z_map, mask, score_threshold, max_p_value, clusters):
    smoothness = subprocess.check_output(["smoothest", "-z", z_map, "-m", mask])
    smoothness = dict(x.split(" ", 1) for x in smoothness.decode().splitlines())
    
    with tempfile.TemporaryDirectory() as directory:
        input = os.path.join(directory, "input.nii.gz")
        output = os.path.join(directory, "output.nii.gz")
        
        command = _get_base_cluster_command(input, score_threshold, output)
        command.extend([
            "--volume={}".format(smoothness["VOLUME"]), 
            "--dlh={}".format(smoothness["DLH"]), 
            "--resels={}".format(smoothness["RESELS"]), "--minclustersize", 
            "--pthresh={}".format(max_p_value)])
        
        clusters_image = _run(command, nibabel.load(z_map), input, output)
        nibabel.save(clusters_image, clusters)

def by_size(score_map, score_threshold, min_size, clusters):
    with tempfile.TemporaryDirectory() as directory:
        input = os.path.join(directory, "input.nii.gz")
        output = os.path.join(directory, "output.nii.gz")
        
        command = _get_base_cluster_command(input, score_threshold, output)
        command.append("--minextent={}".format(min_size))
        
        clusters_image = _run(command, nibabel.load(score_map), input, output)
        nibabel.save(clusters_image, clusters)

def _get_base_cluster_command(source, threshold, output):
    command = [
        "cluster",
        "--in={}".format(source), "--thresh={}".format(threshold), 
        "--othresh={}".format(output)]
    return command

def _run(command, image, input, output):
    data = image.get_fdata()
    
    nibabel.save(nibabel.Nifti1Image(data, image.affine), input)
    subprocess.check_output(command)
    positive = nibabel.load(output).get_fdata()
    
    nibabel.save(nibabel.Nifti1Image(-data, image.affine), input)
    subprocess.check_output(command)
    negative = nibabel.load(output).get_fdata()
    
    return nibabel.Nifti1Image(numpy.abs(positive - negative), image.affine)
