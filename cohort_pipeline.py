import pathlib

import scipy.stats

from exam_pipeline import ExamPipeline
import tasks

# FIXME: hashing Python tasks varies across hosts

def build(sources, transforms, reference, mask, social, atlases, Paths):
    exam_pipelines = {}
    for source in sources:
        subject = str(list(source.parents)[-3])
        exam_pipeline = ExamPipeline(
            Paths(source), transforms[subject], reference)
        exam_pipeline.to_standard
        asymmetry = exam_pipeline.asymmetry

        exam_pipelines[subject] = exam_pipeline
    
    subject_template_links = []
    for exam, exam_pipeline in sorted(exam_pipelines.items()):
        symmetric_template = exam_pipeline.template.targets[-3]
        subject = str(list(exam_pipeline.source.parents)[-3])
        subject_template_links.append(
            tasks.MakeLink(
                pathlib.Path("..")/symmetric_template,
                Paths.cohort_template/(subject.replace("/", "_")+".nii.gz")))
    
    cohort_template = tasks.SymmetricCohortTemplateSlurm(
        [x.targets[0] for x in subject_template_links], reference,
        Paths.cohort_template/"cohort")

    warped_to_cohort = {}
    exam_index = 0
    for index, (exam, exam_pipeline) in enumerate(sorted(exam_pipelines.items())):
        exam_pipeline.cohort_template = cohort_template.targets[-3]
        warped_to_cohort[exam] = cohort_template.targets[4*exam_index+3]
        exam_pipeline.cohort_transforms = [
            cohort_template.targets[4*exam_index+1],
            cohort_template.targets[4*exam_index]]

        asymmetry_to_cohort_template = exam_pipeline.asymmetry_to_cohort_template

        exam_index += 1
    
    groups = [
        sorted([
            p.asymmetry_to_cohort_template.targets[0] 
            for x, p in exam_pipelines.items() if social.get(x) == s])
        for s in [True, False]
    ]
    
    group_comparison = tasks.WelchTest(
        groups, mask, 
        Paths.vba/"t.nii.gz", Paths.vba/"p.nii.gz", Paths.vba/"z.nii.gz")
    
    # Use Student's t-test DoF instead of Welch-Satterthwaite since we would 
    # depend on the standard deviation of the samples
    t = scipy.stats.t(df=sum(len(g) for g in groups)-2)
    min_cluster_size = 100
    min_region_size = 50
    for threshold in [0.01, 0.005, 0.001]:
        clusters = tasks.SizeClustering(
            group_comparison.targets[0], t.ppf(1-threshold), min_cluster_size, 
            Paths.vba/"clusters_t_{}_{}.nii.gz".format(
                threshold, min_cluster_size))
        
        for atlas_name, (atlas_labels, atlas_image) in atlases.items():
            clusters_volume_report = tasks.ClustersVolumeReport(
                clusters.targets[0], atlas_image, atlas_labels, 
                Paths.vba/"clusters_t_{}_{}_{}_{}.xlsx".format(
                    threshold, min_cluster_size, min_region_size, atlas_name), 
                min_size=min_region_size)
    
    groups = [
        sorted([
            image 
            for exam, image in warped_to_cohort.items()
            if social.get(exam) == s])
        for s in [True, False]
    ]
    tasks.AverageImages(groups[0], Paths.vba/"social.nii.gz")
    tasks.AverageImages(groups[1], Paths.vba/"not_social.nii.gz")
