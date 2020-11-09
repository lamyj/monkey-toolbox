import argparse
import sys

import nibabel
import numpy
import pandas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clusters")
    parser.add_argument("atlas")
    parser.add_argument("labels")
    parser.add_argument(
        "volumes", help="Path to the output volumes table, "
        "in CSV or XLS/XLSX format")
    parser.add_argument("--min-size", type=int)
    arguments = parser.parse_args()
        
    clusters = nibabel.load(arguments.clusters)
    atlas = nibabel.load(arguments.atlas)
    voxel_volume = numpy.abs(numpy.linalg.det(atlas.affine[:3,:3]))
    
    with open(arguments.labels) as fd:
        labels_data = fd.read()
    labels = {}
    for line in labels_data.splitlines():
        line = line.strip()
        if not line:
            continue
        number, name = line.split(" ", 1)
        labels[int(number)] = name
    
    mask = (clusters.get_fdata() != 0)
    cluster_labels = numpy.round(atlas.get_fdata()[mask]).astype(int)
    volumes = []
    for number, name in labels.items():
        count = (cluster_labels == number).sum()
        if arguments.min_size and count <= arguments.min_size:
            # print(number, name)
            continue
        volumes.append([name, number, voxel_volume*count])
    volumes = pandas.DataFrame(
        volumes, columns=["Name", "Number", "Volume (mm³)"])
    volumes.sort_values("Volume (mm³)", inplace=True, ascending=False)
    print(volumes)
    
    if arguments.volumes.endswith(".csv"):
        volumes.to_csv(arguments.volumes, index=False)
    elif any(arguments.volumes.endswith(x) for x in [".xls", ".xlsx"]):
        volumes.to_excel(arguments.volumes, index=False)
    else:
        raise Exception("Unknown output format: {}".format(arguments.volumes))

if __name__ == "__main__":
    sys.exit(main())
