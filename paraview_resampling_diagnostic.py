"""Run in ParaView's Python Shell with Run Script (local/builtin session).

Reads existing pipeline outputs only: no UpdatePipeline, Fetch, new filters,
locators, mesh copies, or full-size result arrays. Scalar scans use small chunks.
Reports to the Python Shell. No files or pipeline settings are changed.
"""

from collections import Counter, defaultdict
import numpy as np
from paraview.simple import GetSources
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkVersion, reference
from vtkmodules.vtkCommonDataModel import vtkGenericCell, vtkCellTypes

FINE = "final_res(19).vtu"
ADAPTIVE = "final_res(18).vtu"
RESAMPLED = "ResampleWithDataset1"
DENSITY = "cell_data_1"
CUTOFF = 0.5
CHUNK = 65536
CELL_SAMPLES = 128


def main():
    sources = GetSources()
    by_name = defaultdict(list)
    names_by_id = {}
    for (name, _), proxy in sources.items():
        by_name[name].append(proxy)
        names_by_id[proxy.SMProxy.GetGlobalIDAsString()] = name

    def proxy_named(name):
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise RuntimeError("Expected one source named %r. Available: %s" %
                               (name, ", ".join(sorted(by_name))))
        return matches[0]

    def existing_data(proxy):
        algo = proxy.SMProxy.GetClientSideObject()
        if algo is None or not hasattr(algo, "GetOutputDataObject"):
            raise RuntimeError("Requires a local/builtin session; no Fetch fallback.")
        data = algo.GetOutputDataObject(0)
        if data is None or not data.IsA("vtkDataSet"):
            raise RuntimeError("Expected an already-applied, single dataset.")
        if data.GetNumberOfPoints() == 0:
            raise RuntimeError("Output is empty/not computed. Apply it in the GUI first.")
        return data

    def scalar(data, association, name):
        attrs = data.GetCellData() if association == "cell" else data.GetPointData()
        array = attrs.GetArray(name)
        if array is None or array.GetNumberOfComponents() != 1:
            raise RuntimeError("Missing scalar %s array %r" % (association, name))
        expected = data.GetNumberOfCells() if association == "cell" else data.GetNumberOfPoints()
        if array.GetNumberOfTuples() != expected:
            raise RuntimeError("Array tuple count does not match its dataset.")
        # VTU numeric arrays are exposed as views, not copied wholesale.
        return vtk_to_numpy(array).reshape(-1)

    def connections(proxy):
        for key in ("Input", "SourceDataArrays", "DestinationMesh"):
            prop = proxy.SMProxy.GetProperty(key)
            if prop is not None and prop.IsA("vtkSMInputProperty"):
                names = [names_by_id.get(prop.GetProxy(i).GetGlobalIDAsString(), "?")
                         for i in range(prop.GetNumberOfProxies())]
                print("  %s (ordered): %s" % (key, names))

    def describe(label, data):
        print("\n%s: %s, cells=%d, points=%d" %
              (label, data.GetClassName(), data.GetNumberOfCells(), data.GetNumberOfPoints()))
        print("  bounds:", tuple(round(x, 9) for x in data.GetBounds()))
        for association, attrs in (("cell", data.GetCellData()), ("point", data.GetPointData())):
            print("  %s arrays: %s" % (association,
                  [attrs.GetArrayName(i) for i in range(attrs.GetNumberOfArrays())]))

    def stats(label, values):
        finite_count = solid_count = 0
        low, high = float("inf"), -float("inf")
        for start in range(0, values.size, CHUNK):
            block = values[start:start + CHUNK]
            finite = np.isfinite(block)
            finite_count += int(np.count_nonzero(finite))
            solid_count += int(np.count_nonzero(finite & (block >= CUTOFF)))
            if np.any(finite):
                good = block[finite]
                low, high = min(low, float(good.min())), max(high, float(good.max()))
        print("  %s: finite=%d/%d, range=%s, density >= %g: %d" %
              (label, finite_count, values.size,
               (low, high) if finite_count else "none", CUTOFF, solid_count))

    print("=== BEGIN PARAVIEW RESAMPLING DIAGNOSTIC ===")
    print("VTK:", vtkVersion.GetVTKVersion(), "| cutoff:", CUTOFF)
    print("Existing cached outputs only; apply pending GUI changes before running.")
    fine_proxy, adaptive_proxy, sampled_proxy = map(proxy_named, (FINE, ADAPTIVE, RESAMPLED))
    fine, adaptive, sampled = map(existing_data, (fine_proxy, adaptive_proxy, sampled_proxy))
    for label, data in ((FINE, fine), (ADAPTIVE, adaptive), (RESAMPLED, sampled)):
        describe(label, data)
    print("\nResampling connections:")
    connections(sampled_proxy)
    for name, proxies in by_name.items():
        if name.startswith("PythonCalculator") or name.startswith("CellCenters"):
            for proxy in proxies:
                print("\n%s connections:" % name)
                connections(proxy)
    if adaptive.GetPointData().HasArray(DENSITY):
        print("WARNING: adaptive POINT density also exists; probing can prefer it over CELL density.")

    f = scalar(fine, "cell", DENSITY)
    a = scalar(adaptive, "cell", DENSITY)
    r = scalar(sampled, "point", DENSITY)
    mask = scalar(sampled, "point", "vtkValidPointMask")
    print("\nDensity statistics (exact counts; small temporary chunks):")
    for label, array in (("fine", f), ("adaptive", a), ("sampled adaptive", r)):
        stats(label, array)
    if not (f.size == r.size == mask.size):
        raise RuntimeError("Fine cells, sampled points and mask have different counts; comparison stopped.")

    counts = Counter()
    failed_solid_examples = []
    for start in range(0, f.size, CHUNK):
        fb, rb, mb = (x[start:start + CHUNK] for x in (f, r, mask))
        valid = mb == 1
        usable = valid & np.isfinite(fb) & np.isfinite(rb)
        fs, rs = fb >= CUTOFF, rb >= CUTOFF
        counts["valid probes"] += int(np.count_nonzero(valid))
        counts["invalid probes"] += int(np.count_nonzero(~valid))
        counts["invalid probes inside fine solid"] += int(np.count_nonzero(~valid & fs))
        counts["valid probes with nonfinite density"] += int(np.count_nonzero(valid & ~usable))
        counts["fine solid only (valid)"] += int(np.count_nonzero(usable & fs & ~rs))
        counts["adaptive solid only (valid)"] += int(np.count_nonzero(usable & ~fs & rs))
        counts["solid in both (valid)"] += int(np.count_nonzero(usable & fs & rs))
        if len(failed_solid_examples) < 5:
            ids = np.flatnonzero(~valid & fs)[:5-len(failed_solid_examples)]
            failed_solid_examples.extend(start + int(i) for i in ids)
    print("\nComparison counts (assumes unchanged Cell Centers order):")
    for label, count in counts.items():
        print("  %s: %d (%.4f%% of fine cells)" % (label, count, 100.0 * count / f.size))
    for cid in failed_solid_examples:
        print("  example invalid fine-solid cell %d, probe XYZ=%s" %
              (cid, tuple(round(x, 9) for x in sampled.GetPoint(cid))))

    # Verify the correspondence assumed above on a small, evenly spaced sample.
    fine_cell = vtkGenericCell()
    max_center_error = 0.0
    for cid in np.unique(np.linspace(0, f.size - 1, min(f.size, 32), dtype=np.int64)):
        fine.GetCell(int(cid), fine_cell)
        pc, xyz = [0.0]*3, [0.0]*3
        subid = reference(fine_cell.GetParametricCenter(pc))
        fine_cell.EvaluateLocation(subid, pc, xyz, [0.0]*fine_cell.GetNumberOfPoints())
        probe_xyz = sampled.GetPoint(int(cid))
        max_center_error = max(max_center_error,
                               max(abs(xyz[j]-probe_xyz[j]) for j in range(3)))
    print("\nFine-center / sampled-point coordinate mismatch (32-cell sample):", max_center_error)
    print("  Expected approximately zero. A nonzero mismatch invalidates the paired comparison.")

    # Only inspect a bounded sample. No spatial locator or all-cell geometry array.
    cell = vtkGenericCell()
    groups = defaultdict(Counter)
    n = adaptive.GetNumberOfCells()
    print("\nAdaptive cell geometry: at most %d cells sampled (not exhaustive)." % CELL_SAMPLES)
    for cid in np.unique(np.linspace(0, n - 1, min(n, CELL_SAMPLES), dtype=np.int64)):
        adaptive.GetCell(int(cid), cell)
        b = cell.GetBounds()
        widths = tuple(round(b[2*j+1] - b[2*j], 9) for j in range(3))
        key = (vtkCellTypes.GetClassNameFromTypeId(cell.GetCellType()), widths)
        group = groups[key]
        group["sampled"] += 1
        group["solid"] += int(a[int(cid)] >= CUTOFF)
        group["dimension"] = cell.GetCellDimension()
        # Bounding-box test applies only to nondegenerate axis-aligned hex/voxel cells.
        if cell.GetCellType() not in (11, 12) or min(widths) <= 0:
            continue
        eps = max(widths) * 1e-8
        corners = set()
        for j in range(cell.GetNumberOfPoints()):
            xyz = cell.GetPoints().GetPoint(j)
            corner = []
            for axis in range(3):
                lo, hi = b[2*axis:2*axis+2]
                corner.append(0 if abs(xyz[axis]-lo) <= eps else
                              1 if abs(xyz[axis]-hi) <= eps else -1)
            corners.add(tuple(corner))
        if len(corners) != 8 or any(-1 in c for c in corners):
            continue
        group["axis-aligned cells checked"] += 1
        for fraction in ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)):
            xyz = [b[2*j] + fraction[j] * (b[2*j+1]-b[2*j]) for j in range(3)]
            closest, pcoords = [0.0]*3, [0.0]*3
            subid, dist2 = reference(0), reference(0.0)
            weights = [0.0] * cell.GetNumberOfPoints()
            inside = cell.EvaluatePosition(xyz, closest, subid, pcoords, dist2, weights)
            group["interior tests"] += 1
            group["interior tests rejected"] += int(inside != 1)
    for (kind, widths), values in sorted(groups.items()):
        print("  %s, widths=%s: %s" % (kind, widths, dict(values)))
    print("\nInvalid probes are UNKNOWN, not void. Valid-only counts are incomplete if probes fail.")
    print("=== END PARAVIEW RESAMPLING DIAGNOSTIC ===")


try:
    main()
except Exception:
    import traceback
    print("DIAGNOSTIC STOPPED; no pipeline settings were changed.")
    traceback.print_exc()
