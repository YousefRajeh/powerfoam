from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import inspect
from typing import Optional, Union

import configargparse


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise configargparse.ArgumentTypeError("Boolean value expected. True or False.")


def add_group(parser: configargparse.ArgParser, group_class):
    fields = dict(inspect.getmembers(group_class))["__dataclass_fields__"]
    for name, field in fields.items():
        # A dataclass field with default_factory (needed for mutable defaults such as
        # list[float]) has field.default == MISSING, which would otherwise mark it REQUIRED
        # on the command line. Resolve the factory to get the real default.
        has_factory = field.default_factory is not dataclasses.MISSING
        required = field.default is dataclasses.MISSING and not has_factory
        t = field.type
        nargs = None
        if hasattr(t, "__origin__") and hasattr(t, "__args__"):
            origin = t.__origin__
            if origin is Union and len(t.__args__) == 2 and type(None) in t.__args__:
                for arg in t.__args__:
                    if arg != type(None):
                        t = arg
                        break
                if required:
                    field.default = None
                    required = False
            elif issubclass(origin, Sequence):
                t = t.__args__[0]
                nargs = "+"
            else:
                raise ValueError(f"Unsupported type {t} for field {name}")

        if required and t == bool:
            parser.add_argument(
                f"--{name}", type=str_to_bool, nargs="?", const=True, required=True
            )
        elif required:
            parser.add_argument(f"--{name}", nargs=nargs, type=t, required=True)
        elif t == bool:
            assert field.default == False
            parser.add_argument(f"--{name}", action="store_true")
        else:
            default = field.default_factory() if has_factory else field.default
            parser.add_argument(f"--{name}", nargs=nargs, type=t, default=default)

    def extract(args):
        kwargs = {}
        for arg in vars(args).items():
            if arg[0] in fields:
                kwargs[arg[0]] = arg[1]
        return group_class(**kwargs)

    return extract


@dataclass(kw_only=True)
class Params:
    # Experiement parameters
    iterations: int
    normal_weight: float
    contribution_weight: float
    interpenetration_weight: float
    densify_from: int
    densify_until: int
    # Surface-concentration regulariser (OFF by default -- 0.0 leaves training bit-identical
    # to every run made before it existed). Motivated by VoroTracing (arXiv 2608.17682),
    # which applies Mip-NeRF 360's distortion loss to a Voronoi field and reports that the
    # learned density becomes "strongly bimodal -- cells are either near-transparent or
    # near-opaque". We want that property for a SEMANTIC reason, not a rendering one: ~90% of
    # our cells are interior non-owners, which is why every adjacency-growing method we tried
    # blobbed through object interiors (geodesic FPS -5 mIoU, coherence-gated growth -12.3
    # with 3/10 catastrophic collapses). If opacity is bimodal, "is this cell on a surface"
    # becomes a trained property instead of a threshold we have to guess per scene.
    #
    # NOT literally Mip-NeRF 360's L_dist: that needs per-sample weights w_k = T_k*alpha_k
    # inside the warp ray kernel. This penalises the INTERQUANTILE DEPTH SPREAD instead --
    # the distance between the depths at which transmittance crosses two quantiles, which is
    # the width of the interval holding most of the ray's weight. Driving it to zero forces
    # transmittance to fall 1 -> 0 abruptly, i.e. the same surface-concentrated opacity, and
    # it is differentiable through the renderer's EXISTING depth-quantile backward pass.
    # VoroTracing (arXiv 2608.17682) Sec 5.4: sigma = exp(rho) instead of softplus. The
    # segment length cancels from the gradient, so cells of equal opacity are optimized
    # equally regardless of size. Their reported symptom of the softplus bias -- "large
    # cells that should be empty settle at a low but non-zero density" -- is our
    # interior-non-owner problem. "softplus" (default) keeps the historical behaviour.
    density_activation: str = "softplus"
    # Linear warmup on the density LR. 1000 was hardcoded historically; VoroTracing uses
    # 2000 and it is the only group they warm. Exposed so the two can be A/B'd, since the
    # exponential parameterization is far more sensitive to an early overshoot.
    density_warmup_steps: int = 1_000
    # "iqr"   = interquantile-spread surrogate. FALSIFIED as a substitute for L_dist: it is
    #           a trimmed dispersion measure, provably blind to a 10% tail at ANY distance
    #           (mass 0.9 at s=0 and 0.1 at s=D gives IQR 0 for every D while the true
    #           Gini-mean-difference term grows as 0.18*D). Kept only as a cheap diagnostic.
    # "exact" = the real Mip-NeRF 360 L_dist, evaluated on stratified transmittance
    #           quantiles where the bin masses are exact constants and only the depths are
    #           rendered. Verified against the O(K^2) definition to 1e-7.
    distortion_mode: str = "iqr"
    distortion_num_quantiles: int = 16
    # WHERE the distortion gradient is allowed to act. This is the power-diagram-native
    # question: VoroTracing routes it to DENSITY ONLY, because a midpoint bisector cannot
    # move without moving a site. Our weights translate a cell's planes without moving its
    # center, so "radii" lets the loss mean "make this cell thinner" instead of only "make
    # this material more opaque". "both" is the full channel; "density" reproduces theirs.
    distortion_channel: str = "both"
    distortion_weight: float = 0.0
    # MUST be list[float], never tuple: train.py dumps vars(args) with yaml.dump, and a
    # tuple serializes as `!!python/tuple`, which configargparse cannot read back. That
    # silently breaks EVERY script that reloads a checkpoint's config.yaml (load_foam and
    # therefore every eval in this project). A list round-trips as an inline flow sequence.
    distortion_quantiles: list[float] = dataclasses.field(
        default_factory=lambda: [0.1, 0.9])
    distortion_min_alpha: float = 0.5
    experiment_name: str = ""
    dry_run: bool = False
    viewer: bool = False
    # Enable external normal supervision.  When False (default), the only
    # normal-related term in the loss is the renderer's internal
    # ``normal_err`` regulariser.  When True, an additional MSE supervision
    # term is added against either Metric3D normals (if ``use_metric3d`` is
    # also True) or finite-difference normals computed from the rendered
    # depth map.
    normal_supervision: bool = False

    # Dataset parameters
    dataset: str
    data_path: str
    scene: str
    alpha_format_on_disk: str
    downsample: list[int]
    downsample_iterations: list[int]
    use_metric3d: bool = False
    is_pinhole: bool = False
    eval: bool = False

    # Model parameters
    init_type: str
    init_points: int
    final_points: int
    bkgd_color: list[float]
    disable_coop_prim_load: bool = False
    disable_coop_adj_load: bool = False
    render_objective: str = "volume"
    sv_dof: int
    num_texel_sites: int

    # Optimizer parameters
    points_lr_init: float
    points_lr_final: float
    density_lr_init: float
    density_lr_final: float
    radii_lr_init: float
    radii_lr_final: float
    quaternions_lr_init: float
    quaternions_lr_final: float
    texel_sites_lr_init: float
    texel_sites_lr_final: float
    texel_sv_axis_lr_init: float
    texel_sv_axis_lr_final: float
    texel_sv_rgb_lr_init: float
    texel_sv_rgb_lr_final: float
    texel_height_lr_init: float
    texel_height_lr_final: float
