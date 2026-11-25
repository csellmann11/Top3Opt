# Does HYPRE's OpenMP threading actually use the cores of this job?
# Solves a 3-D Poisson problem with PCG + BoomerAMG and reports how many cores
# were busy (process CPU time / wall time) during setup and solve.
#
#   OMP_NUM_THREADS=16 julia --startup-file=no --project=. cluster/check_hypre_threads.jl        # HYPRE alone
#   OMP_NUM_THREADS=16 julia --startup-file=no --project=. cluster/check_hypre_threads.jl mkl    # MKL loaded first, as in the driver
# Optional second argument: grid points per direction (default 110, i.e. 1.33 million unknowns).
const WITH_MKL = "mkl" in ARGS
WITH_MKL && @eval using MKL
using HYPRE, SparseArrays, LinearAlgebra, Libdl, Printf
HYPRE.Init(; nthreads = parse(Int, get(ENV, "HYPRE_NUM_THREADS", "0")))   # 0: OMP_NUM_THREADS decides; plain HYPRE.Init() forces 1
const L = HYPRE.LibHYPRE

sizes = filter(!isnothing, tryparse.(Int, ARGS))
n = isempty(sizes) ? 110 : first(sizes)
T = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
E = sparse(1.0I, n, n)
A = kron(E, E, T) + kron(E, T, E) + kron(T, E, E)
rhs = ones(size(A, 1))

cpu() = Sys.islinux() ? ccall(:clock, Clong, ()) / 1e6 : NaN     # process CPU seconds, all threads
function timed(f)
    c = cpu(); t = time_ns(); f()
    wall = (time_ns() - t) / 1e9
    return wall, (cpu() - c) / wall
end

gomp = filter(contains("libgomp"), Libdl.dllist())
omp_max = isempty(gomp) ? -1 : Int(ccall(Libdl.dlsym(Libdl.dlopen(first(gomp)), :omp_get_max_threads), Cint, ()))
println("MKL loaded first: ", WITH_MKL, "   OMP_NUM_THREADS=", get(ENV, "OMP_NUM_THREADS", "(unset)"),
        "   OMP_PROC_BIND=", get(ENV, "OMP_PROC_BIND", "(unset)"))
println("libgomp: ", isempty(gomp) ? "not loaded" : first(gomp), "   omp_get_max_threads: ", omp_max, "   HYPRE.NumThreads(): ", HYPRE.NumThreads())
println("OpenMP runtimes loaded: ", join(basename.(filter(l -> occursin(r"gomp|iomp|libomp", l), Libdl.dllist())), ", "))
Sys.islinux() && println("CPUs this process may use (nproc): ", strip(read(`nproc`, String)))
@printf("unknowns: %d   nonzeros: %d\n", size(A, 1), nnz(A))

for rep in 1:2
    Ah = HYPRE.HYPREMatrix(A); b = HYPRE.HYPREVector(rhs); x = HYPRE.HYPREVector(zeros(length(rhs)))
    pre = HYPRE.BoomerAMG(; CoarsenType = 10, RelaxType = 6, NumSweeps = 1, MaxIter = 1, Tol = 0.0)
    pcg = HYPRE.PCG(; MaxIter = 200, Tol = 1e-8, Precond = pre)
    ts, cs = timed(() -> L.@check L.HYPRE_ParCSRPCGSetup(pcg, Ah, b, x))
    tv, cv = timed(() -> L.HYPRE_ParCSRPCGSolve(pcg, Ah, b, x))
    @printf("run %d: setup %.2fs (%.1f cores busy)   solve %.2fs, %d iterations (%.1f cores busy)\n",
            rep, ts, cs, tv, HYPRE.GetNumIterations(pcg), cv)
    foreach(finalize, (pcg, pre, x, b, Ah))
end
