import os
import re

import spire

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
        
        original_stem = re.sub(r'.nii(?:.gz)?', '', os.path.basename(original))
        mirrored_stem = re.sub(r'.nii(?:.gz)?', '', os.path.basename(mirrored))
        
        self.targets = [
            f"{prefix}{original_stem}00GenericAffine.mat",
            f"{prefix}{original_stem}01Warp.nii.gz",
            f"{prefix}{original_stem}01InverseWarp.nii.gz",
            f"{prefix}template0{original_stem}0WarpedToTemplate.nii.gz",
            
            f"{prefix}{mirrored_stem}10GenericAffine.mat",
            f"{prefix}{mirrored_stem}11Warp.nii.gz",
            f"{prefix}{mirrored_stem}11InverseWarp.nii.gz",
            f"{prefix}template0{mirrored_stem}1WarpedToTemplate.nii.gz",
            
            f"{prefix}template0.nii.gz",
            f"{prefix}template0GenericAffine.mat",
            f"{prefix}template0warp.nii.gz"
        ]
        
        # WARNING: ITK_GLOBAL_NUMBER_OF_THREADS is not re-exported by the Slurm
        # scripts, and antsMultivariateTemplateConstruction2.sh removes *all*
        # slurm-*.out in output dir. Use sequential mode.
        self.actions = [
            [
                "antsMultivariateTemplateConstruction2.sh",
                "-d", "3", "-r", "1", "-n", "0",
                "-o", prefix, original, mirrored],
            ["rm", f"{prefix}templatewarplog.txt"]]

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
