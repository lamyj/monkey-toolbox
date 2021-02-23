import nibabel
import numpy
import scipy.stats

def test(groups, mask, t_map, p_map, z_map):
    images = [[nibabel.load(x) for x in g] for g in groups]
    if mask is not None:
        mask = nibabel.load(mask)
        # numpy mask is 1 for invalid values
        mask = numpy.asarray(mask.dataobj)==0
        image_data = [
            [numpy.ma.array(x.get_fdata(), mask=mask) for x in g] for g in images]
        image_data = [numpy.array([x.compressed() for x in g]).T for g in image_data]
    else:
        image_data = [numpy.array([x.get_fdata().ravel() for x in g]).T for g in images]
    
    image = images[0][0]
    arrays = list(scipy.stats.ttest_ind(*image_data, axis=1, equal_var=False))
    arrays.append(-scipy.stats.norm.ppf(0.5*arrays[1]) * numpy.sign(arrays[0]))
    
    for index, flat_array in enumerate(arrays):
        if mask is not None:
            array = numpy.zeros(image.shape, flat_array.dtype)
            array[~mask] = flat_array
        else:
            array = flat_array.reshape(image.shape)
        arrays[index] = array
    
    t, p, z = arrays
    nibabel.save(nibabel.Nifti1Image(t, image.affine), t_map)
    nibabel.save(nibabel.Nifti1Image(p, image.affine), p_map)
    nibabel.save(nibabel.Nifti1Image(z, image.affine), z_map)
