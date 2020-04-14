import re
import subprocess

def adjust_voxel_size(input, output, target_size=1):
    """ Adjust the input voxel size isotropically by an integer multiple so 
        that the smallestsize element is larger than target_size. This is only 
        a header modification, no resampling takes place. 
        
        :param input: path to the input file
        :param output: path to the output file
        :param target_size: target voxel size in mm.
    """
    
    data = subprocess.check_output(["PrintHeader", input, "1"])
    voxel_size = [float(x) for x in re.findall(b"([\d.]+)", data)]
    factor = round(target_size/min(voxel_size))
    subprocess.check_call(
        ["SetSpacing", "3", input, output]
        +[str(x*factor) for x in voxel_size])

def bet(input, output, write_mask=False, threshold=0.5, cog=None):
    """ Run FSL's BET. If ``cog`` is None, the robust version of BET will run.
        Since ``cog`` is given in voxels, it is unaffected by the voxel size
        adjustment.
    """
    
    command = ["bet", input, output]
    command += ["-f", str(threshold)]
    if write_mask:
        command += ["-m"]
    if cog:
        x, y, z = cog
        command += ["-c", str(x), str(y), str(z)]
    else:
        command += ["-R"]
    
    subprocess.check_call(command)

def restore_voxel_size(input, reference):
    """ Restore the voxel size of the input file, based on the reference file.
    """
    
    data = subprocess.check_output(["PrintHeader", reference, "1"])
    voxel_size = [float(x) for x in re.findall(b"([\d.]+)", data)]
    subprocess.check_call(
        ["SetSpacing", "3", input, input]+[str(x) for x in voxel_size])
