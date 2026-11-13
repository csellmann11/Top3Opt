"""Global baseline coefficient; h is the regularization length cbrt(cell volume).
To restore the mesh-size floor r²_i >= 3h_i^2, replace `base` below with `max(base,3h^2)`.
Sensitivity-based local_max scaling is applied separately after this choice.
"""
@inline regularization_coefficient(base,h) = base
