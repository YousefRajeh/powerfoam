# PowerFoam renderer — complete specification

Reconstructed by reading every line of `powerfoam/rendering_math.py` (334), `powerfoam/rasterize.py`
(2646), `powerfoam/texture.py` (302), `powerfoam/camera.py` (342), `powerfoam/raytrace.py` (311),
`powerfoam/geometry.py` (321), plus the adjacency construction in `benchmark.py` and the parameter
activations in `powerfoam/scene.py`.

Verified against upstream `github.com/theialab/powerfoam`: `rendering_math.py` is **byte-identical**;
`rasterize.py` differs only by our additions (`export_operator_kernel`, and the `feature_sim` /
`feature_pca` / `front_prim_idx` / `front_t_surf` / `front_t_entry` outputs on the visualisation
path). The dipole clipping appears in all four upstream kernels (lines 719, 858, 1085, 1362).

---

## 1. The representation

Each primitive `i` carries:

| symbol | storage | meaning |
|---|---|---|
| `c_i` | `points` (P,3) | site position |
| `r_i` | `get_radii()` | bounding-sphere radius **and** the power weight |
| `n_i` | `get_normals()` | dipole plane normal (from `quaternions`) |
| `σ_i` | `get_density()` | volume density |
| `s_ik` | `texel_sites` (P,S,2) | S texel sites, 2-D in the cell's tangent frame |
| `h_ik` | `texel_height` (P,S) | per-texel height offset |
| `rgb_ik` | `texel_rgb` (P,S,3) | per-texel colour |

Packed for the kernels as `all_spheres = [c, r]` (vec4f) and `all_nsigmas = [n, σ]` (vec4).
Texel sites are lifted to **world** space before launch:

```
offsets     = texel_sites * r[:, None, None]
texel_world = points[:, None, :] + offsets[...,0:1]*tangent + offsets[...,1:2]*bitangent
```

`tangent`/`bitangent` are columns 2 and 3 of the rotation matrix of `quaternions`
(`get_tangents`), and `n_i` is column 1 (`get_normals`), so the frame is orthonormal.

Density activation (`scene.py::get_density`) is `σ = exp(ρ)` (VoroTracing parameterisation), chosen
because `dL/dρ = dL/dα · (1−α) · ln(1/(1−α))` — the **segment length cancels**, so cells of any size
get equal gradient at equal opacity. Under softplus it does not cancel and small cells are
under-trained; the paper attributes floaters and haze to exactly that.

## 2. Adjacency: a Čech-filtered regular triangulation

`benchmark.py::build_adjacency` / `scene.py::rebuild_adjacency`:

1. lift each site to 4-D as `(c_i, ‖c_i‖² − r_i²)`
2. convex hull; keep faces with `equations[:,3] < 0` (the **lower** hull) — this is exactly the
   **regular (weighted Delaunay) triangulation**
3. expand tetrahedra to their 6 edges, dedupe
4. **if `alpha_complex` (TRUE for the rasteriser): keep only edges with `‖c_i − c_j‖ < r_i + r_j`**
   — only pairs whose bounding spheres overlap. This is the alpha/Čech complex, a **strict subset**
   of the regular triangulation.
5. symmetrise into CSR `adjacency` + `adjacency_offsets`

**This matters — see §8.1.**

`prefetch_adjacency_kernel` precomputes per directed edge `vec4h(c_j − c_i, r_j)` into
`adjacency_diff` (fp16). `benchmark.py` packs the *same array* differently — slot 3 there is
`pm_j − pm_i` with `pm = ½(‖c‖² − r²)`, i.e. the power-face **offset**. Two packings, two consumers,
each self-consistent:

| packing | slot 3 | consumed by |
|---|---|---|
| `prefetch_adjacency_kernel` | `r_j` | `ray_pface_intersect` (full form) — forward, visualisation, export_operator |
| `benchmark.py` | `pm_j − pm_i` | `ray_pface_intersect_diff` — benchmark + raytrace kernels |

## 3. Culling and ordering

**Per-primitive reject** (both cull kernels): `if ‖v‖ < 4r or dot(v, forward) < 0.1: return`, with
`v = c − eye`, `forward = cross(up, right)`. So the eye is never within 4 radii of a kept sphere,
which guarantees `pow_dist = ‖v‖² − r² ≥ 15r² > 0` (used in §3.2).

### 3.1 Tile assignment — two paths, selected by `args.is_pinhole`

**Pinhole.** `proj_sphere_to_obb` projects the sphere to an NDC ellipse by solving the conic
coefficients `a,b,c,d,e,f` from `C = c − eye`, extracting eigenvalues
`l1,l2 = (a+b ± √((a−b)²+c²))/2` and semi-axes `r1,r2 = √(−F0/l)`. The OBB's AABB bounds a tile
loop; each candidate tile is confirmed by `verify_tile_obb_intersection`, a 2-axis
separating-axis test against the OBB's major/minor axes.

**Generic.** A **cone hierarchy** over the ray map. `compute_leaf_cones_kernel` builds a cone per
`LEAF_SIZE × LEAF_SIZE` (2×2) pixel block: axis = normalised mean ray, `cos_half` = min dot to member
rays. `merge_cones_kernel` merges 2×2 children with the exact spherical bound
`extent = cosθ·cos_half_child − sinθ·sin_half_child`. `cos_half > 1.0` is the empty sentinel.
Traversal uses an explicit 32-deep stack; `sphere_cone_intersect` is
`cos_alpha > cos_half·cos_beta − sin_half·sin_beta` with `sin_beta = r/dist`.

Both run **twice**: `count_visible_*` fills `tile_inter_counts`, a `cumsum` gives `offsets`, then
`write_visible_*` scatters `prim_indices` via an atomic per-tile cursor.

### 3.2 Ordering

```
sort_keys[idx] = (tile_idx << 32) | uint32_bits(pow_dist)
tile_prim_indices = tile_prim_indices[argsort(sort_keys)]
```

`wp.cast(float, uint32)` is a **bit reinterpretation** (verified: 0.25 → 1048576000 = 0x3E800000),
and IEEE-754 patterns are monotone in value for **non-negative** floats. The `4r` cull guarantees
`pow_dist > 0`, so the sort is exact at full float precision.

But `pow_dist` is a **per-primitive** scalar shared by all 64 rays of a tile, while true entry depth
is per-ray — so traversal is only **approximately** front-to-back. That is the origin of the measured
85–90% `front_prim_idx` agreement and the ~7% α-ordering violations. It is the same approximation
3DGS makes, not a defect.

## 4. The per-cell segment: three clips

Per candidate primitive in tile order, ray `(o, d)` with `d` **normalised** (so `t` is metric):

**(a) Bounding sphere.** `ray_sphere_intersect` → `[t_near, t_far]`, `t_near` clamped to 0 if the eye
is inside. Miss ⇒ `continue`.

**(b) Power facets — pure sign tests.** For each Čech neighbour `j`:

```
face_n      = c_j − c_i
face_offset = ½(‖c_j‖² − ‖c_i‖² + r_i² − r_j²)
dp = dot(d, face_n);   t_face = (face_offset − dot(o, face_n)) / dp
t_far  = min(t_face, t_far)   if dp ≥ 0
t_near = max(t_face, t_near)  if dp <  0
if t_near > t_far: break        # ray misses this cell
```

Setting `pow(x,i) = pow(x,j)` with `pow(x,k) = ‖x−c_k‖² − r_k²` gives exactly
`face_n · x = face_offset`. Membership is therefore a **conjunction of linear inequalities**
`face_n · x − face_offset ≤ 0`; the `t` merely locates where the sign flips along the ray.

**(c) The dipole.** `plane_intersection_fwd[_local]` intersects an oriented plane through `c_i` with
normal `n_i`, displaced by a spatially varying height:

```
t_surf₀, dp₀ = ray_plane_intersect(o, d, c_i, n_i)              # h = 0
q₀           = o + (t_near if dp₀≥0 else max(t_near,t_surf₀))·d
h(q)         = Σ_k w_k h_ik / Σ_k w_k,   w_k = exp(−10·‖q − s_ik‖²/r_i²)
t_surf, dp   = ray_plane_intersect(o, d, c_i, n_i, h(q₀))       # displaced
colour       = same Gaussian-weighted mean of rgb_ik at the NEW intersection point
```

— one fixed-point step linearising a curved height field per ray (`temp = 10`, `max_sites = 8`). Then

```
t_far  = min(t_surf, t_far)   if dp ≥ 0
t_near = max(t_surf, t_near)  if dp <  0
```

`dp = dot(n_i, d)`, and **both branches keep the same half**: occupied ⇔ `(x − c_i)·n_i < h(x)`.
Only one half of the cell holds matter — the side the normal points away from.

**(d) Composite.**

```
dt = t_far − t_near
α  = 1 − exp(−σ_i·dt)          # density_integral is homogeneous: −σ·dt
w  = α·exp(log_t)
rgb   += colour·w
log_t += −σ_i·dt
```

Early-out when `exp(log_t) < transmittance_threshold` (default 1e-3). In `forward_kernel` the test is
`wp.tile_max` over the tile, so all 64 pixels stop **together**, and the stopping index is recorded in
`tile_early_stop_counter` for the backward replay.

## 5. Kernel inventory

| kernel | purpose | facet form | notes |
|---|---|---|---|
| `count_visible_pinhole/generic` | tile histogram | — | 4r + frustum cull |
| `write_visible_pinhole/generic` | scatter + sort key | — | atomic per-tile cursor |
| `compute_leaf_cones` / `merge_cones` | cone BVH | — | generic cameras only |
| `prefetch_adjacency` | pack `(Δc, r_j)` fp16 | — | per directed edge |
| `benchmark_kernel` | timing only | `_diff` | consumes `benchmark.py` packing |
| `visualization_kernel` | viewer + our feature/front outputs | full | depth = **quantile** |
| `export_operator_kernel` *(ours)* | sparse `A[pixel,prim] = α·T` | full | geometry identical to above |
| `forward_kernel` | training | full | `wp.tile` coop loads, multi-quantile, `contrib_out` |
| `backward_kernel` | training | full | reverse replay |

## 6. Backward pass

Replays the tile list in **reverse** (`early_stop−1 → 0`), recovering transmittance by
`log_t −= delta_log_t`. It tags which entity set each bound —
`t_near_id`/`t_far_id ∈ {−2 none, −1 sphere, 0..n_adj−1 facet, n_adj dipole}` — and routes `dL/dt`
accordingly: `ray_sphere_intersect_bwd`, `ray_pface_intersect_bwd` (which **also writes into the
neighbour's** sphere gradient, since a facet depends on both sites), or `plane_intersection_bwd`.
Gradients land on `points`, `radii`, `density`, `normals`, `texel_sites`, `texel_rgb`,
`texel_height`. Non-finite gradients are zeroed host-side. Surface-loss accumulation
(`render_objective == "surface"`) raises `NotImplementedError` in backward.

## 7. Outputs

`visualize()` returns 10 tensors; the last three are ours and are **appended** (callers index
positionally: `[1]` depth, `[3]` alpha):

```
color, depth, normal, alpha, intersections, feature_heat, feature_pca,
front_prim_idx, front_t_surf, front_t_entry
```

`forward()` returns `color, opacity = 1−exp(log_t), normal_distance, normal, quantile_depths, err,
contrib, point_err, prim_visible_mask`. `normal_distance` accumulates `(n·d)²·α·T` only where
`n·d > 0` (back-facing).

## 8. Defects and sharp edges found while reading

**8.1 The rasteriser clips against a SUBSET of facets — measured, and it is negligible.** The
alpha-complex filter keeps only neighbours whose bounding spheres overlap, so the clipped region is
a **superset** of the true sphere-clipped power cell. Measured with `measure_cech_partition.py`
(uniform samples in each cell's bounding sphere; membership by exact `argmin` power distance over
**all** P sites, no k-NN shortcut):

| arm | r max/min | stolen (of sampled volume) | stolen / rendered | prims affected | lost |
|---|---|---|---|---|---|
| truefrozen | 3.75×10⁵ | **0.05%** | 0.11% | 1.5% | **0.0%** |
| nonfrozen | 27.2 | **0.02%** | 0.02% | 0.5% | **0.0%** |

`lost = 0.0%` is the built-in correctness check: a point owned by `i` satisfies every half-space and
so must satisfy the Čech subset. The leak is at numerical-edge-case level even with radii spanning
375,000×.

**The partition property does not depend on this measurement.** It is a property of the
PARAMETERISATION: a power diagram assigns every point of R³ to exactly one cell by definition of
`argmin` power distance, and no triangulation quirk can express "two cells in the same place". A
clipping leak means the renderer shades marginally beyond the mathematical boundary; it does not
make foam a mixture. By contrast **two 3DGS Gaussians with identical means and covariances are a
perfectly valid configuration** — overlap is *inexpressible* in one representation and
*unconstrained* in the other. That is the structural form of the A34 argument, and it is stronger
than the empirical CV of 1.2%.

**8.2 `visualize()` is pinhole-only.** It calls `count_visible_kernel`/`write_visible_kernel` with the
pinhole argument list unconditionally, but `__init__` binds the generic cone-hierarchy variants when
`args.is_pinhole` is false.

**8.3 `visualize()`'s default `VisOptions` never sets `depth_quantile`.** It sets
`transmittance_threshold`, `max_intersections`, `bkgd_color` only, so `depth_quantile` defaults to 0,
`if next_trans < options.depth_quantile` is never true, and **`depth_out` is returned all zeros**.
Any caller using `result[1]` as depth gets zeros. Use `front_t_surf`/`front_t_entry`, or pass an
explicit quantile.

**8.4 `texture.py` defines `soft_voronoi_fwd`/`_bwd` twice each** (float and vec3f overloads); in
Python the second rebinds the first. Unused by the rasteriser, but misleading.

**8.5 The two `adjacency_diff` packings are undocumented at the call sites** (§2). Feeding the wrong
packing to a consumer produces silently wrong geometry rather than an error.

## 9. Consequences for the lifting work

* `export_operator_kernel` reproduces the forward geometry exactly, so the operator `A` behind every
  oracle result **is** the renderer's own compositing weights.
* The renderer's "inside cell i" is **sphere ∩ power-facet half-spaces ∩ dipole half-space** —
  strictly smaller than `assign_points_to_power_cells`, which implements only the power-cell term.
  Measured: **19.2%** (truefrozen) / **41.2%** (nonfrozen) of GT points sit on the *empty* side of
  their owning cell's dipole.
* Slot order in the exported operator is tile order (power-distance sorted per tile), hence only
  approximately depth-ordered. Any front/back split built on slot index inherits that error.
* Depth-render check vs the GT mesh: `truefrozen` `|t_surf − t_mesh|` median **2.0 cm**;
  `nonfrozen` median **17.0 cm**. Unfrozen geometry sits far off the true surface, which explains its
  per-primitive collapse, its 41.2% dipole-empty points, and its low `front_agree` (0.821 vs 0.967).

---

## 10. Oracle GT mesh provenance (closed)

`scenes10_points3d/<scene>/points3d.ply` **is the official ScanNet `<scene>_vh_clean_2.ply`,
byte-identical on all 10 scenes**, verified with `cmp` against
`http://kaldir.vc.in.tum.de/scannet/v2/scans/<id>/<id>_vh_clean_2.ply`:

| scene | bytes | | scene | bytes |
|---|---|---|---|---|
| scene0000_00 | 3,298,819 | | scene0347_00 | 2,761,856 |
| scene0062_00 | 2,118,072 | | scene0400_00 | 6,438,607 |
| scene0070_00 | 4,456,523 | | scene0590_00 | 9,129,567 |
| scene0097_00 | 2,940,468 | | scene0645_00 | 14,459,000 |
| scene0140_00 | 15,307,815 | | scene0200_00 | 3,414,953 |

The `comment VCGLIB generated` header line is ScanNet's own, not a re-save by us.

Corroborating, from independent copies:
* official `_vh_clean_2.labels.ply` (NormLift_release): vertex counts match on all 10 scenes, and
  its per-vertex labels agree with our `segment20` remap at **100.0000%** on all 10.
* a Kaggle SparseConvNet `.pth` copy: coords max|d| = 0, colours 100.0000%, labels 100.0000%.

**Note for anyone repeating this:** every redistribution of ScanNet is FACE-STRIPPED (Pointcept
`.npy`, NormLift `.labels.ply`, Kaggle `.pth`, HuggingFace `frames_square`) because the 3D
segmentation community works on point clouds. Only the official `_vh_clean_2.ply` download carries
connectivity, and connectivity is what the oracle needs for visibility.
