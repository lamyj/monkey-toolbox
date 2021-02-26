import inspect
import spire.ants
import tasks

class ExamPipeline(object):
    def __init__(self, paths, transforms, reference):
        self.source = paths.source
        self.destination = paths.destination
        
        self.transforms = transforms
        self.reference = reference
        
        self.cohort_template = None
        self.cohort_transforms = None
        
        self._tasks = {}
    
    @property
    def to_standard(self):
        return self._get_task(
            tasks.ManualTransform, 
            self.source, self.transforms, self.reference,
            self.destination/"to_standard.txt")
    
    @property
    def reorientation(self):
        return self._get_task(
            tasks.Reorient,
            self.source, self.transforms, self.to_standard.targets[0], 
            self.reference, self.destination/"reoriented.nii.gz")
    
    @property
    def preprocessing(self):
        return self._get_task(
            tasks.BiasCorrection,
            self.reorientation.targets[0], 
            self.destination/"reoriented_preprocessed.nii.gz")
    
    @property
    def mirroring(self):
        return self._get_task(
            tasks.Mirror,
            self.preprocessing.targets[0], 
            self.destination/"mirrored_preprocessed.nii.gz")

    @property
    def template(self):
        return self._get_task(
            tasks.SymmetricSubjectTemplate,
            self.preprocessing.targets[0], self.mirroring.targets[0],
            self.destination/"symmetric")
    
    @property
    def original_jacobian(self):
        # NOTE https://sourceforge.net/p/advants/discussion/840260/thread/84d24a38/
        # Differences between geometric Jacobian and finite differences Jacobian
        # are minimal
        return self._get_task(
            tasks.JacobianDeterminant,
            self.template.targets[1], 
            self.destination/"original_jacobian.nii.gz",
            True)
    
    @property
    def mirrored_jacobian(self):
        return self._get_task(
            tasks.JacobianDeterminant,
            self.template.targets[5], 
            self.destination/"mirrored_jacobian.nii.gz",
            True)

    @property
    def asymmetry(self):
        return self._get_task(
            tasks.Subtract,
            self.original_jacobian.targets[0], self.mirrored_jacobian.targets[0], 
            self.destination/"asymmetry.nii.gz")
    
    @property
    def asymmetry_to_cohort_template(self):
        return self._get_task(
            spire.ants.ApplyTransforms,
            self.asymmetry.targets[0], 
            self.cohort_template, self.cohort_transforms,
            self.destination/"asymmetry_in_cohort_template.nii.gz")
    
    ############################################################################
    #                             Private interface                            #
    ############################################################################
    
    def _get_task(self, class_, *args, **kwargs):
        name = inspect.stack()[1][3]
        if name not in self._tasks:
            self._tasks[name] = class_(*args, **kwargs)
        return self._tasks[name]
    
    def _jacobian(self, source, target_prefix):
        
        return self._get_task(tasks.JacobianDeterminant, source, target, True)
