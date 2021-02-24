import nibabel
import numpy
import pandas
import yaml

def clusters_volume_report(clusters, atlas, labels, report, min_size=None):
    clusters = nibabel.load(clusters)
    atlas = nibabel.load(atlas)
    labels = yaml.load(open(labels), yaml.CLoader)
    
    mask = (clusters.get_fdata() != 0)
    cluster_labels = numpy.round(atlas.get_fdata()[mask]).astype(int)
    
    voxel_volume = numpy.abs(numpy.linalg.det(atlas.affine[:3,:3]))
    
    volumes = []
    for number, name in labels.items():
        count = (cluster_labels == number).sum()
        if min_size and count <= min_size:
            # print(number, name)
            continue
        volumes.append([name, number, voxel_volume*count])
    
    volumes = pandas.DataFrame(
        volumes, columns=["Name", "Number", "Volume (mm³)"])
    volumes.sort_values("Volume (mm³)", inplace=True, ascending=False)
    
    if report.endswith(".csv"):
        volumes.to_csv(report, index=False)
    elif report.endswith(".xlsx"):
        volumes.to_excel(report, index=False)
    else:
        raise Exception("Unknown output format: {}".format(report))
