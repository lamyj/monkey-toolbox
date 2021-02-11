import re

import spire

class Reorient(spire.TaskFactory):
    def __init__(self, source, image_transform, grid_transforms, target):
        spire.TaskFactory.__init__(self, str(target))
        self.file_dep = [source]
        self.targets = [target]
        
        self.actions = []
        if image_transform is not None:
            self.actions.append(
                ["same-grid-transform", source, image_transform, target])
        else:
            self.actions.append(["cp", source, target])
        self.actions.extend([[x, target, target] for x in grid_transforms])

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
