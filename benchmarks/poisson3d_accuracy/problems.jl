isdefined(@__MODULE__,:gauss_legendre01) || include("quadrature.jl")

@inline poisson_interval_factor(t) = t*(1-t)

@inline function poisson_cube_exact(point)
    x,y,z = point
    return poisson_interval_factor(x)*poisson_interval_factor(y)*poisson_interval_factor(z)
end

@inline function poisson_cube_gradient(point)
    x,y,z = point
    X,Y,Z = poisson_interval_factor(x),poisson_interval_factor(y),poisson_interval_factor(z)
    return ((1-2x)*Y*Z,X*(1-2y)*Z,X*Y*(1-2z))
end

@inline function poisson_cube_rhs(point)
    x,y,z = point
    X,Y,Z = poisson_interval_factor(x),poisson_interval_factor(y),poisson_interval_factor(z)
    return 2(Y*Z+X*Z+X*Y)
end

@inline function poisson_lshape_smooth_exact(point)
    x,y,z = point
    return x*(1-x*x)*y*(1-y*y)*poisson_interval_factor(z)
end

@inline function poisson_lshape_smooth_gradient(point)
    x,y,z = point
    X,Y,Z = x*(1-x*x),y*(1-y*y),poisson_interval_factor(z)
    return ((1-3x*x)*Y*Z,X*(1-3y*y)*Z,X*Y*(1-2z))
end

@inline function poisson_lshape_smooth_rhs(point)
    x,y,z = point
    X,Y,Z = x*(1-x*x),y*(1-y*y),poisson_interval_factor(z)
    return 6(x*Y+y*X)*Z+2X*Y
end

"""Wedge singularity s=r^(2/3)sin(2theta/3) and its Cartesian derivatives.

The domain angle is 0 <= theta <= 3pi/2. Derivatives are evaluated only away
from the reentrant edge; no finite gradient is assigned to that singularity.
"""
@inline function poisson_wedge_field(x,y)
    r = hypot(x,y)
    r > 0 || throw(DomainError((x,y),"The reentrant-edge gradient is undefined"))
    theta = mod(atan(y,x),2pi)
    alpha = 2/3
    s = ((y == 0 && x >= 0) || (x == 0 && y <= 0)) ? 0.0 :
        r^alpha*sin(alpha*theta)
    derivative = alpha*r^(alpha-1)
    return s,derivative*sin((alpha-1)*theta),derivative*cos((alpha-1)*theta)
end

@inline function poisson_lshape_singular_exact(point)
    x,y,z = point
    iszero(x) && iszero(y) && return 0.0
    s,_,_ = poisson_wedge_field(x,y)
    return (1-x*x)*(1-y*y)*s*poisson_interval_factor(z)
end

@inline function poisson_lshape_singular_gradient(point)
    x,y,z = point
    Z = poisson_interval_factor(z)
    if iszero(x) && iszero(y)
        # At the two end points the extra vanishing z factor gives zero first
        # derivatives; the interior of the reentrant edge remains singular.
        iszero(Z) && return (0.0,0.0,0.0)
        throw(DomainError(point,"The manufactured gradient is singular at the reentrant edge"))
    end
    s,sx,sy = poisson_wedge_field(x,y)
    F = (1-x*x)*(1-y*y)
    Fx,Fy = -2x*(1-y*y),-2y*(1-x*x)
    return ((Fx*s+F*sx)*Z,(Fy*s+F*sy)*Z,F*s*(1-2z))
end

@inline function poisson_lshape_singular_rhs(point)
    x,y,z = point
    iszero(x) && iszero(y) && return 0.0
    s,sx,sy = poisson_wedge_field(x,y)
    F = (1-x*x)*(1-y*y)
    Fx,Fy = -2x*(1-y*y),-2y*(1-x*x)
    laplacian_F = -2(1-y*y)-2(1-x*x)
    return -poisson_interval_factor(z)*(s*laplacian_F+2(Fx*sx+Fy*sy))+2F*s
end

"""Independent polar integration of the singular problem's reference energy.

Integrate z analytically: integral Z²=1/30 and integral (Z')²=1/3. Splitting
theta at multiples of pi/4 resolves each branch of the square's radial bound.
The substitution r=R(theta)*t³ removes the fractional radial singularity.
"""
function poisson_singular_reference_energy(order::Integer)
    nodes,weights = gauss_legendre01(order)
    energy = 0.0
    angle_width = pi/4
    @inbounds for segment in 0:5,i in eachindex(nodes)
        theta = (segment+nodes[i])*angle_width
        cosine,sine = cos(theta),sin(theta)
        radius = 1/max(abs(cosine),abs(sine))
        for j in eachindex(nodes)
            t = nodes[j]
            r = radius*t^3
            x,y = r*cosine,r*sine
            s,sx,sy = poisson_wedge_field(x,y)
            F = (1-x*x)*(1-y*y)
            hx = -2x*(1-y*y)*s+F*sx
            hy = -2y*(1-x*x)*s+F*sy
            h = F*s
            radial_jacobian = 3radius^2*t^5
            density = 0.5*((hx*hx+hy*hy)/30+h*h/3)
            energy += weights[i]*weights[j]*angle_width*radial_jacobian*density
        end
    end
    return energy
end

"""Manufactured unit-diffusion Poisson problem with homogeneous Dirichlet data.

`domain=:cube` denotes [0,1]^3. `domain=:lshape` denotes
((−1,1)^2 minus {x>0,y<0}) × (0,1). Gradient callbacks return 3-tuples.
`reference_uncertainty` is an order-convergence estimate for the singular case,
not a mathematically certified error bound; polynomial references are analytic.
"""
function manufactured_problem(name::Symbol)
    if name == :cube_smooth
        return (;name,domain=:cube,exact=poisson_cube_exact,
            gradient=poisson_cube_gradient,rhs=poisson_cube_rhs,
            reference_energy=1/1800,reference_uncertainty=0.0,
            description="Smooth product polynomial on the unit cube; unit diffusion and zero boundary values.")
    elseif name == :lshape_smooth
        energy = (3/8)*(2*(8/5)*(16/105)*(1/30)+(16/105)^2/3)
        return (;name,domain=:lshape,exact=poisson_lshape_smooth_exact,
            gradient=poisson_lshape_smooth_gradient,rhs=poisson_lshape_smooth_rhs,
            reference_energy=energy,reference_uncertainty=0.0,
            description="Smooth polynomial on the extruded three-quadrant L-shaped domain; zero boundary values.")
    elseif name == :lshape_singular
        coarse = poisson_singular_reference_energy(32)
        energy = poisson_singular_reference_energy(48)
        uncertainty = max(abs(energy-coarse),64eps(Float64)*max(abs(energy),abs(coarse)))
        return (;name,domain=:lshape,exact=poisson_lshape_singular_exact,
            gradient=poisson_lshape_singular_gradient,rhs=poisson_lshape_singular_rhs,
            reference_energy=energy,reference_uncertainty=uncertainty,
            description="Reentrant-edge singularity r^(2/3)sin(2theta/3) times a boundary-vanishing polynomial envelope.")
    end
    throw(ArgumentError("Unknown manufactured problem $name; choose :cube_smooth, :lshape_smooth, or :lshape_singular"))
end
