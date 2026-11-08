"""Global baseline coefficient; h is the cell diameter.
To restore the mesh-size floor, replace `base` below with `max(base,h^2)`.
Sensitivity-based local_max scaling is applied separately after this choice.
"""
@inline regularization_coefficient(base,h) = base
