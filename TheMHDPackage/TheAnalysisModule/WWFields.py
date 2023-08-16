## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
def vectorCrossProduct(vector1, vector2):
  vector3 = np.array([
    vector1[1] * vector2[2] - vector1[2] * vector2[1],
    vector1[2] * vector2[0] - vector1[0] * vector2[2],
    vector1[0] * vector2[1] - vector1[1] * vector2[0]
  ])
  return vector3

def vectorDotProduct(vector1, vector2):
  scalar = np.sum([
    v1_comp * v2_comp
    for v1_comp, v2_comp in zip(vector1, vector2)
  ], axis=0)
  return scalar

def fieldMagnitude(vector_field):
  vector_field = np.array(vector_field)
  return np.sqrt(np.sum(vector_field**2, axis=0))

def gradient_2ocd(field, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  return (
    np.roll(field, F, axis=gradient_dir) - np.roll(field, B, axis=gradient_dir)
  ) / (2*cell_width)

def fieldRMS(scalar_field):
  return np.sqrt(np.mean(scalar_field**2))

def fieldGradient(scalar_field):
  ## format: (x, y, z)
  scalar_field = np.array(scalar_field)
  cell_width = 1 / scalar_field.shape[0]
  field_gradient = [
    gradient_2ocd(scalar_field, cell_width, gradient_dir)
    for gradient_dir in [0, 1, 2]
  ]
  return np.array(field_gradient)

def computeTNBBasis(vector_field):
  ## format: (component, x, y, z)
  vector_field = np.array(vector_field)
  field_magn = fieldMagnitude(vector_field)
  ## ---- COMPUTE TANGENT BASIS
  t_basis = vector_field / field_magn
  ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  gradient_tensor = np.array([
    fieldGradient(field_component)
    for field_component in vector_field
  ])
  ## ---- COMPUTE NORMAL BASIS
  ## f_i df_j/dx_i
  n_basis_term1 = np.einsum("ixyz,jixyz->jxyz", vector_field, gradient_tensor)
  ## f_i f_j f_m df_m/dx_i
  n_basis_term2 = np.einsum("ixyz,jxyz,mxyz,mixyz->jxyz", vector_field, vector_field, vector_field, gradient_tensor)
  ## (f_i df_j/dx_i) / (f_k f_k) - (f_i f_j f_m df_m/dx_i) / (f_k f_k)^2
  n_basis = n_basis_term1 / field_magn**2 - n_basis_term2 / field_magn**4
  ## field curvature
  kappa = fieldMagnitude(n_basis)
  ## normal basis
  n_basis /= kappa
  ## ---- COMPUTE BINORMAL BASIS
  ## orthogonal to both t- and b-basis
  b_basis = vectorCrossProduct(t_basis, n_basis)
  return t_basis, n_basis, b_basis, kappa


## END OF LIBRARY