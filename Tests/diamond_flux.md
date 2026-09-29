# 3D density regularization

The default is now `--flux_scheme diamond`. For comparisons, select
`--flux_scheme tpfa` or `--flux_scheme strong`. The Julia entry point accepts
`run_optimization(...; flux_scheme=:diamond)`. `laplace_rescale` applies only to
the legacy `strong` scheme. Output directory identifiers include the flux scheme.

The face schemes assemble `div(beta_hat * grad(chi))`, with the existing 3D
coefficient `beta_hat[i] = 2 * beta0 * h[i]^2`. The density update multiplies this
by `p_avg` once. The operator is rebuilt after mesh adaptation, so both geometry
and the spatially varying coefficient follow the current mesh.

Each active interior subface contributes one flux with opposite signs to its
two neighboring cells. Its transmissibility is
`area / (delta_left/beta_left + delta_right/beta_right)`. Diamond adds the two
tangential gradient components reconstructed from face vertex values. Vertex
values use local linear least squares and mirrored samples on boundary faces,
following the 2D reference. Boundary flux is zero.

The operator preserves constants and conserves the volume-weighted total.
It reduces to TPFA on orthogonal faces. Like the reference correction, it is not
guaranteed symmetric or monotone; the density update still clamps densities and
enforces the volume constraint. An absolute-row-sum bound increases the explicit
substep count when needed, but is not a general stability proof for arbitrary
distorted meshes. Planar faces and cell centers on opposite sides of each
interior face are required.

Run the focused regression tests with:

```powershell
julia --project=. --startup-file=no Tests/diamond_flux.jl
```

They check two-material flux with a discontinuous normal gradient, tangential
correction, orientation independence, uniform and locally refined meshes,
constant preservation, conservation, nodal affine reconstruction away from the
boundary, and zero regularization. They do not replace an optimization benchmark.
