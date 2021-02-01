import numpy

def transform_matrix(source, transform):
    # Since P = D⋅I + O (P: physical coordinates, I: index coordinates, 
    # D: direction matrix, O: origin, i.e. physical coordinate of I=(0,0,0)), 
    # given a transformed T centered on C, its action on a point P is
    # T⋅(P - C) + C = T⋅(D⋅I + O) - T⋅C + C = (T⋅D)⋅I + T⋅(O-C) + C
    # The new direction matrix is hence T⋅D, and the new origin is T⋅(O-C) + C.

    # Physical coordinates of image center
    center = source.affine @ [*numpy.divide(source.shape, 2.), 1.]
    center = center[:3]/center[3]

    # Original direction and offset
    direction = source.affine[:3, :3]
    origin = source.affine[:3, 3]

    # Build the new affine matrix with previous formula
    affine = numpy.full_like(source.affine, 0.)
    affine[:3, :3] = transform @ direction
    affine[:3, 3] = transform @ (origin - center) + center
    affine[3, 3] = 1
    
    return affine

def axis_angle_to_matrix(axis, angle):
    r""" Convert an (axis, angle) to a rotation matrix.
    
         This formula comes from Rodrigues' rotation formula,
         :math:`R = I + \hat{\omega} \sin \theta + \hat{\omega}^2 (1-\cos \theta)`
         where the :math:`\hat{}` operator gives the antisymmetric matrix 
         equivalent of the cross product:
        
         .. math ::
            
             \hat{\omega} = \left(\begin{matrix}
                0         & -\omega_z &  \omega_y \\
                \omega_z  &         0 & -\omega_x \\
                -\omega_y &  \omega_x &         0 \\
            \end{matrix}\right)
        
         Diagonal terms can be rewritten:
         
         .. math ::
             
             \begin{matrix}
                 1+(1-\cos \theta)*(\omega_x^2-1) & = & 1+(1-\cos \theta)*\omega_x^2-(1-\cos \theta) \\
                                                  & = & \cos \theta+\omega_x^2*(1-\cos \theta)
             \end{matrix}
    """
    
    result = numpy.ndarray((3,3))
    
    cos = numpy.cos(angle)
    sin = numpy.sin(angle)
    one_minus_cos = 1.-cos
    
    result[0][0] = cos+axis[0]**2*(one_minus_cos)
    result[1][1] = cos+axis[1]**2*(one_minus_cos)
    result[2][2] = cos+axis[2]**2*(one_minus_cos)
    
    result[0][1] = -axis[2]*sin+axis[0]*axis[1]*one_minus_cos
    result[1][0] = +axis[2]*sin+axis[0]*axis[1]*one_minus_cos
    
    result[0][2] = +axis[1]*sin+axis[0]*axis[2]*one_minus_cos
    result[2][0] = -axis[1]*sin+axis[0]*axis[2]*one_minus_cos
    
    result[1][2] = -axis[0]*sin+axis[1]*axis[2]*one_minus_cos
    result[2][1] = +axis[0]*sin+axis[1]*axis[2]*one_minus_cos
    
    return result
