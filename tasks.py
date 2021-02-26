import csv
import itertools
import os
import re

import spire
import yaml

import clustering
import clusters_volume_report
import welch

class ManualTransform(spire.TaskFactory):
    def __init__(self, source, grid_transforms, reference, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source, reference]
        self.targets = [target]
        self.actions = (
            [
                ["echo", "Save transform in", target],
                ["cp", source, "tmp.nii.gz"]]
            + [[x, "tmp.nii.gz", "tmp.nii.gz"] for x in grid_transforms]
            + [
                ["itksnap", "-g", reference, "-o", "tmp.nii.gz"],
                ["rm", "tmp.nii.gz"]])

class Reorient(spire.TaskFactory):
    def __init__(self, source, grid_transforms, image_transform, reference, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        
        self.actions = [["cp", source, target]]
        self.actions.extend([[x, target, target] for x in grid_transforms])
        if image_transform is not None:
            self.actions.append(
                [
                    "antsApplyTransforms", 
                    "-i", target, "-r", reference, "-t", image_transform,
                    "-o", target])

class BiasCorrection(spire.TaskFactory):
    def __init__(self, source, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [["N4BiasFieldCorrection", "-i", source, "-o", target]]

class Mirror(spire.TaskFactory):
    def __init__(self, source, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [["lr-mirror", source, target]]

class SymmetricSubjectTemplate(spire.TaskFactory):
    def __init__(self, original, mirrored, prefix):
        spire.TaskFactory.__init__(self, str(prefix))
        self.file_dep = [original, mirrored]
        
        original_stem = re.sub(
            r".nii(?:.gz)?", "", os.path.basename(str(original)))
        mirrored_stem = re.sub(
            r".nii(?:.gz)?", "", os.path.basename(str(mirrored)))
        
        self.targets = [
            "{}{}00GenericAffine.mat".format(prefix, original_stem),
            "{}{}01Warp.nii.gz".format(prefix, original_stem),
            "{}{}01InverseWarp.nii.gz".format(prefix, original_stem),
            "{}template0{}0WarpedToTemplate.nii.gz".format(prefix, original_stem),
            
            "{}{}10GenericAffine.mat".format(prefix, mirrored_stem),
            "{}{}11Warp.nii.gz".format(prefix, mirrored_stem),
            "{}{}11InverseWarp.nii.gz".format(prefix, mirrored_stem),
            "{}template0{}1WarpedToTemplate.nii.gz".format(prefix, mirrored_stem),
            
            "{}template0.nii.gz".format(prefix),
            "{}template0GenericAffine.mat".format(prefix),
            "{}template0warp.nii.gz".format(prefix)
        ]
        
        # WARNING: ITK_GLOBAL_NUMBER_OF_THREADS is not re-exported by the Slurm
        # scripts, and antsMultivariateTemplateConstruction2.sh removes *all*
        # slurm-*.out in output dir. Use sequential mode.
        self.actions = [
            [
                "antsMultivariateTemplateConstruction2.sh",
                "-d", "3", "-r", "1", "-n", "0",
                "-o", prefix, original, mirrored],
            ["rm", "{}templatewarplog.txt".format(prefix)]]

class JacobianDeterminant(spire.TaskFactory):
    def __init__(self, source, target, log=False, geometric=False):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [
            [
                "CreateJacobianDeterminantImage", "3",
                source, target, str(int(log)), str(int(geometric))
            ]]

class Subtract(spire.TaskFactory):
    def __init__(self, lhs, rhs, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [lhs, rhs]
        self.targets = [target]
        self.actions = [["ImageMath", "3", target, "-", lhs, rhs]]

class MakeLink(spire.TaskFactory):
    def __init__(self, source, target):
        spire.TaskFactory.__init__(self, str(target))
        
        # NOTE: source is a path relative to target
        true_source = os.path.relpath(
            os.path.abspath(os.path.join(os.path.dirname(target), source)))
        self.file_dep = [true_source]
        
        self.targets = [target]
        self.actions = [
            ["mkdir", "-p", os.path.dirname(target)],
            ["ln", "-s", "-f", source, target]]

class SymmetricCohortTemplateSlurm(spire.TaskFactory):
    def __init__(self, sources, initial_template, prefix):
        spire.TaskFactory.__init__(self, str(prefix))
        self.file_dep = sources+[initial_template]
        
        self.targets = []
        for i, source in enumerate(sources):
            stem = re.sub(r'.nii(?:.gz)?', '', os.path.basename(source))
            self.targets.extend([
                "{}{}{}0GenericAffine.mat".format(prefix, stem, i),
                "{}{}{}1Warp.nii.gz".format(prefix, stem, i),
                "{}{}{}1InverseWarp.nii.gz".format(prefix, stem, i),
                "{}template0{}{}WarpedToTemplate.nii.gz".format(prefix, stem, i)])
        
        self.targets.extend([
            "{}template0.nii.gz".format(prefix),
            "{}template0GenericAffine.mat".format(prefix),
            "{}template0warp.nii.gz".format(prefix)
        ])
        
        # WARNING: ITK_GLOBAL_NUMBER_OF_THREADS is not re-exported by the Slurm
        # scripts, and antsMultivariateTemplateConstruction2.sh removes *all*
        # slurm-*.out in output dir.
        self.actions = [
            [
                "antsMultivariateTemplateConstruction2.sh",
                "-d", "3", "-r", "1", "-n", "0", "-z", initial_template,
                "-c", "5", "-v", "64G",
                "-o", prefix]+sources,
            ["rm", "{}templatewarplog.txt".format(prefix)]]

class WelchTest(spire.TaskFactory):
    def __init__(self, groups, mask, t_map, p_map, z_map):
        spire.TaskFactory.__init__(self, str(t_map))
        self.file_dep = list(itertools.chain(*groups, [mask]))
        
        self.targets = [t_map, p_map, z_map]
        self.actions = [(welch.test, (groups, mask, t_map, p_map, z_map))]

class SizeClustering(spire.TaskFactory):
    def __init__(self, source, threshold, min_size, target):
        spire.TaskFactory.__init__(self, str(target))
        
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [
            (clustering.by_size, (source, threshold, min_size, target))]

class PValueClustering(spire.TaskFactory):
    def __init__(self, source, mask, threshold, max_p_value, target):
        spire.TaskFactory.__init__(self, str(target))
        
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [
            (
                clustering.by_p_value, 
                (source, mask, threshold, max_p_value, target))]

class D99Labels(spire.TaskFactory):
    """ Parse the D99 label map to a standard format. """
    
    def __init__(self, source, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [(D99Labels.parse_labels, (source, target))]
    
    @staticmethod
    def parse_labels(source, target):
        with open(source) as fd:
            labels_data = fd.read()
        labels = {}
        for line in labels_data.splitlines():
            line = line.strip()
            if not line:
                continue
            number, name = line.split(" ", 1)
            labels[int(number)] = name
        with open(target, "w") as fd:
            yaml.dump(labels, fd)

class CHARMLabels(spire.TaskFactory):
    """ Parse the CHARM/SARM label maps to a standard format. """
    
    def __init__(self, source, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        self.actions = [(CHARMLabels.parse_labels, (source, target))]
    
    @staticmethod
    def parse_labels(source, target):
        with open(source) as fd:
            reader = csv.DictReader(fd, delimiter="\t")
            labels_data = list(reader)
        labels = {}
        for entry in labels_data:
            labels[int(entry["Index"])] = entry["Full_Name"]
        with open(target, "w") as fd:
            yaml.dump(labels, fd)

class ClustersVolumeReport(spire.TaskFactory):
    def __init__(self, clusters, atlas, labels, report, min_size=None):
        spire.TaskFactory.__init__(self, str(report))
        self.file_dep = [clusters, atlas, labels]
        self.targets = [report]
        self.actions = [
            (
                clusters_volume_report.clusters_volume_report, 
                (clusters, atlas, labels, report, min_size))
        ]
