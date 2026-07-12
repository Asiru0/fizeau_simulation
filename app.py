from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fizeau_sim.metrics import foreground_balanced_psnr, gradient_similarity, psnr, ssim
from fizeau_sim.physical import PhysicalOptics, mtf_from_centered_otf, otf_for_wavelength
from fizeau_sim.reconstruct import multi_pose_soft_adaptive_wiener, wiener_filter
from fizeau_sim.simulate import add_gaussian_noise, apply_otf, make_test_target


ARCSEC_PER_RAD = 206264.80624709636
IMAGE_SIZE = 256


@dataclass(frozen=True)
class PoseInput:
    centers_mm: tuple[tuple[float, float], ...]
    aperture_diameter_mm: float
    focal_length_mm: float
    detector_side_mm: float
    spectrum: tuple[tuple[float, float], ...]
    noise_sigma: float
    regularization: float
    seed: int | None


@dataclass(frozen=True)
class PoseSpec:
    name: str
    array_name: str
    rotation_deg: float
    manual_weight: float
    noise_sigma: float
    seed: int | None
    enabled: bool


@dataclass(frozen=True)
class MultiPoseInput:
    arrays: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
    aperture_diameter_mm: float
    focal_length_mm: float
    detector_side_mm: float
    spectrum: tuple[tuple[float, float], ...]
    default_noise_sigma: float
    base_seed: int | None
    poses: tuple[PoseSpec, ...]
    fusion_method: str
    weight_mode: str
    regularization: float
    mtf_threshold: float
    transition_width: float


def _parse_float(value: str, label: str, minimum: float | None = None) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} 必须是有限数字")
    if minimum is not None and parsed <= minimum:
        raise ValueError(f"{label} 必须大于 {minimum:g}")
    return parsed


def _parse_optional_int(value: str, label: str) -> int | None:
    if not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} 必须是整数或留空") from exc


def _parse_nonnegative_float(value: str, label: str) -> float:
    parsed = _parse_float(value, label)
    if parsed < 0.0:
        raise ValueError(f"{label} 不能为负数")
    return parsed


def _parse_optional_float_token(value: str, label: str) -> float | None:
    if value.strip().lower() in {"", "-", "default"}:
        return None
    return _parse_nonnegative_float(value, label)


def _parse_spectrum(value: str) -> tuple[tuple[float, float], ...]:
    spectrum: list[tuple[float, float]] = []
    for line_number, line in enumerate(value.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(f"光谱第 {line_number} 行必须是：波长(nm) 相对强度")
        wavelength = _parse_float(fields[0], f"光谱第 {line_number} 行波长", 0.0)
        intensity = _parse_float(fields[1], f"光谱第 {line_number} 行相对强度")
        if intensity < 0.0:
            raise ValueError(f"光谱第 {line_number} 行相对强度不能为负数")
        spectrum.append((wavelength, intensity))
    if not spectrum:
        raise ValueError("光谱至少需要一行有效数据")
    total_intensity = sum(intensity for _, intensity in spectrum)
    if total_intensity <= 0.0:
        raise ValueError("光谱相对强度总和必须大于 0")
    return tuple(
        (wavelength, intensity / total_intensity) for wavelength, intensity in spectrum
    )


def _parse_pose_plan(
    value: str,
    array_names: set[str],
    default_noise_sigma: float,
    base_seed: int | None,
) -> tuple[PoseSpec, ...]:
    poses: list[PoseSpec] = []
    names: set[str] = set()
    for line_number, line in enumerate(value.splitlines(), start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        fields = content.replace(",", " ").split()
        if len(fields) != 7:
            raise ValueError(
                f"姿态计划第 {line_number} 行需要 7 列：name array angle weight noise seed enabled"
            )
        name, array_name, angle, weight, noise, seed, enabled = fields
        if name in names:
            raise ValueError(f"姿态名称重复：{name}")
        if array_name not in array_names:
            raise ValueError(f"姿态 {name} 引用了不存在的阵列构型：{array_name}")
        manual_weight = _parse_nonnegative_float(weight, f"姿态 {name} 权重")
        noise_value = _parse_optional_float_token(noise, f"姿态 {name} 噪声")
        if seed.lower() in {"-", "default"}:
            seed_value = None if base_seed is None else base_seed + len(poses)
        else:
            seed_value = _parse_optional_int(seed, f"姿态 {name} 随机种子")
        if enabled.lower() in {"1", "true", "yes", "on"}:
            is_enabled = True
        elif enabled.lower() in {"0", "false", "no", "off"}:
            is_enabled = False
        else:
            raise ValueError(f"姿态 {name} 的 enabled 必须是 1 或 0")
        poses.append(
            PoseSpec(
                name=name,
                array_name=array_name,
                rotation_deg=_parse_float(angle, f"姿态 {name} 旋转角"),
                manual_weight=manual_weight,
                noise_sigma=default_noise_sigma if noise_value is None else noise_value,
                seed=seed_value,
                enabled=is_enabled,
            )
        )
        names.add(name)
    enabled_poses = tuple(pose for pose in poses if pose.enabled)
    if not enabled_poses:
        raise ValueError("姿态计划至少需要一个 enabled=1 的姿态")
    return enabled_poses


def _parse_array_configs(
    values: dict[str, str],
    configs: dict[str, list[dict[str, str | int]]],
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    parsed: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for name, apertures in configs.items():
        if not apertures:
            raise ValueError(f"阵列构型 {name} 至少需要一个子孔径")
        centers = tuple(
            (
                _parse_float(
                    values.get(
                        f"multi_{name}_{aperture['id']}_x",
                        st.session_state.get(
                            f"multi_{name}_{aperture['id']}_x", str(aperture["x"])
                        ),
                    ),
                    f"{name} P{index} X",
                ),
                _parse_float(
                    values.get(
                        f"multi_{name}_{aperture['id']}_y",
                        st.session_state.get(
                            f"multi_{name}_{aperture['id']}_y", str(aperture["y"])
                        ),
                    ),
                    f"{name} P{index} Y",
                ),
            )
            for index, aperture in enumerate(apertures, start=1)
        )
        parsed.append((name, centers))
    return tuple(parsed)


def make_multi_pose_input(
    values: dict[str, str],
    configs: dict[str, list[dict[str, str | int]]],
) -> MultiPoseInput:
    arrays = _parse_array_configs(values, configs)
    default_noise = _parse_nonnegative_float(values["multi_noise"], "高斯噪声")
    base_seed = _parse_optional_int(values["multi_base_seed"], "基础随机种子")
    poses = _parse_pose_plan(
        values["pose_plan"], {name for name, _ in arrays}, default_noise, base_seed
    )
    regularization = _parse_nonnegative_float(
        values["multi_regularization"], "融合正则化系数"
    )
    mtf_threshold = _parse_nonnegative_float(values["multi_mtf_threshold"], "MTF 阈值")
    if mtf_threshold > 1.0:
        raise ValueError("MTF 阈值必须在 0 到 1 之间")
    transition_width = _parse_float(values["multi_transition_width"], "软过渡宽度", 0.0)
    return MultiPoseInput(
        arrays=arrays,
        aperture_diameter_mm=_parse_float(values["multi_diameter"], "统一孔径口径", 0.0),
        focal_length_mm=_parse_float(values["multi_focal"], "有效焦距", 0.0),
        detector_side_mm=_parse_float(values["multi_detector"], "探测器边长", 0.0),
        spectrum=_parse_spectrum(values["multi_spectrum"]),
        default_noise_sigma=default_noise,
        base_seed=base_seed,
        poses=poses,
        fusion_method=values["fusion_method"],
        weight_mode=values["weight_mode"],
        regularization=regularization,
        mtf_threshold=mtf_threshold,
        transition_width=transition_width,
    )


def make_pose_input(values: dict[str, str], aperture_ids: list[int]) -> PoseInput:
    if not aperture_ids:
        raise ValueError("至少需要一个子孔径")
    centers = tuple(
        (
            _parse_float(values[f"x{index}"], f"孔径 {index} X"),
            _parse_float(values[f"y{index}"], f"孔径 {index} Y"),
        )
        for index in aperture_ids
    )
    return PoseInput(
        centers_mm=centers,
        aperture_diameter_mm=_parse_float(values["diameter"], "统一孔径口径", 0.0),
        focal_length_mm=_parse_float(values["focal"], "有效焦距", 0.0),
        detector_side_mm=_parse_float(values["detector"], "探测器边长", 0.0),
        spectrum=_parse_spectrum(values["spectrum"]),
        noise_sigma=_parse_float(values["noise"], "高斯噪声相对强度"),
        regularization=_parse_float(values["regularization"], "Wiener 正则化系数"),
        seed=_parse_optional_int(values["seed"], "噪声随机种子"),
    )


def _params(config: PoseInput) -> PhysicalOptics:
    return PhysicalOptics(
        aperture_diameter_m=config.aperture_diameter_mm * 1e-3,
        focal_length_m=config.focal_length_mm * 1e-3,
        detector_pixel_m=config.detector_side_mm * 1e-3 / IMAGE_SIZE,
        image_size=IMAGE_SIZE,
    )


@st.cache_data(show_spinner=False)
def _compute_optics(
    centers_mm: tuple[tuple[float, float], ...],
    aperture_diameter_mm: float,
    focal_length_mm: float,
    detector_side_mm: float,
    spectrum: tuple[tuple[float, float], ...],
) -> dict[str, np.ndarray | float]:
    config = PoseInput(
        centers_mm=centers_mm,
        aperture_diameter_mm=aperture_diameter_mm,
        focal_length_mm=focal_length_mm,
        detector_side_mm=detector_side_mm,
        spectrum=spectrum,
        noise_sigma=0.0,
        regularization=0.0,
        seed=None,
    )
    params = _params(config)
    centers = np.asarray(config.centers_mm, dtype=float) * 1e-3
    target = make_test_target(IMAGE_SIZE)
    monochromatic_otfs = [
        otf_for_wavelength(centers, params, wavelength * 1e-9)
        for wavelength, _ in config.spectrum
    ]
    weights = np.asarray([intensity for _, intensity in config.spectrum], dtype=float)
    weights /= float(np.sum(weights))
    otf = np.tensordot(weights, np.stack(monochromatic_otfs, axis=0), axes=(0, 0))
    center = IMAGE_SIZE // 2
    otf = otf / otf[center, center]
    mtf = mtf_from_centered_otf(otf)
    degraded = apply_otf(target, otf)
    fov_arcsec = config.detector_side_mm / config.focal_length_mm * ARCSEC_PER_RAD
    scale_arcsec_per_mm = ARCSEC_PER_RAD / config.focal_length_mm
    frequency_limit_cycles_per_mm = params.nyquist_frequency_cpm / 1e3
    return {
        "target": target,
        "otf": otf,
        "mtf": mtf,
        "degraded": degraded,
        "fov_arcsec": fov_arcsec,
        "scale_arcsec_per_mm": scale_arcsec_per_mm,
        "frequency_limit_cycles_per_mm": frequency_limit_cycles_per_mm,
        "array_centers_mm": centers * 1e3,
    }


def run_simulation(config: PoseInput) -> dict[str, np.ndarray | float]:
    result = _compute_optics(
        config.centers_mm,
        config.aperture_diameter_mm,
        config.focal_length_mm,
        config.detector_side_mm,
        config.spectrum,
    ).copy()
    noisy = add_gaussian_noise(
        result["degraded"], sigma=config.noise_sigma, seed=config.seed
    )
    result["noisy"] = noisy
    result["reconstruction"] = wiener_filter(
        noisy, result["otf"], noise_to_signal=config.regularization
    )
    return result


def _rotate_centers(
    centers_mm: tuple[tuple[float, float], ...], angle_deg: float
) -> tuple[tuple[float, float], ...]:
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
    )
    rotated = np.asarray(centers_mm, dtype=float) @ rotation.T
    return tuple((float(x), float(y)) for x, y in rotated)


def _pose_weights(config: MultiPoseInput) -> np.ndarray:
    if config.weight_mode == "equal":
        weights = np.ones(len(config.poses), dtype=float)
    elif config.weight_mode == "inverse_noise":
        variances = np.asarray([pose.noise_sigma**2 for pose in config.poses], dtype=float)
        weights = 1.0 / np.maximum(variances, 1e-12)
    else:
        weights = np.asarray([pose.manual_weight for pose in config.poses], dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("参与融合的姿态权重必须全部大于 0")
    return weights / float(np.mean(weights))


def _weighted_multi_pose_wiener(
    observations: list[np.ndarray],
    otfs: list[np.ndarray],
    weights: np.ndarray,
    regularization: float,
    mtf_threshold: float,
) -> np.ndarray:
    numerator = np.zeros_like(otfs[0], dtype=np.complex128)
    denominator = np.zeros_like(otfs[0], dtype=float)
    for observed, otf, weight in zip(observations, otfs, weights):
        spectrum = np.fft.fftshift(np.fft.fft2(observed))
        numerator += weight * np.conj(otf) * spectrum
        denominator += weight * np.abs(otf) ** 2
    composite_mtf = np.sqrt(denominator)
    composite_mtf /= max(float(composite_mtf.max()), 1e-12)
    estimate_spectrum = numerator / (denominator + regularization)
    estimate_spectrum *= composite_mtf >= mtf_threshold
    return np.real(np.fft.ifft2(np.fft.ifftshift(estimate_spectrum)))


def run_multi_pose_simulation(
    config: MultiPoseInput,
) -> dict[str, np.ndarray | float | list[str]]:
    arrays = dict(config.arrays)
    observations: list[np.ndarray] = []
    otfs: list[np.ndarray] = []
    pose_names: list[str] = []
    target: np.ndarray | None = None
    first_observation: np.ndarray | None = None
    for pose in config.poses:
        rotated_centers = _rotate_centers(arrays[pose.array_name], pose.rotation_deg)
        optics = _compute_optics(
            rotated_centers,
            config.aperture_diameter_mm,
            config.focal_length_mm,
            config.detector_side_mm,
            config.spectrum,
        )
        observation = add_gaussian_noise(
            optics["degraded"], sigma=pose.noise_sigma, seed=pose.seed
        )
        target = optics["target"]
        if first_observation is None:
            first_observation = observation
        observations.append(observation)
        otfs.append(optics["otf"])
        pose_names.append(pose.name)
    weights = _pose_weights(config)
    if config.fusion_method == "standard":
        reconstruction = _weighted_multi_pose_wiener(
            observations,
            otfs,
            weights,
            config.regularization,
            config.mtf_threshold,
        )
    else:
        reconstruction = multi_pose_soft_adaptive_wiener(
            observations,
            otfs,
            noise_variances=list(1.0 / weights),
            base_regularization=config.regularization,
            mtf_threshold=config.mtf_threshold,
            transition_width=config.transition_width,
        )
    denominator = np.zeros_like(otfs[0], dtype=float)
    for otf, weight in zip(otfs, weights):
        denominator += weight * np.abs(otf) ** 2
    composite_mtf = np.sqrt(denominator)
    composite_mtf /= max(float(composite_mtf.max()), 1e-12)
    fov_arcsec = config.detector_side_mm / config.focal_length_mm * ARCSEC_PER_RAD
    frequency_limit = 1.0 / (
        2.0 * (config.detector_side_mm / IMAGE_SIZE)
    )
    assert target is not None and first_observation is not None
    return {
        "target": target,
        "first_observation": first_observation,
        "reconstruction": reconstruction,
        "composite_mtf": composite_mtf,
        "pose_names": pose_names,
        "pose_count": float(len(config.poses)),
        "fov_arcsec": fov_arcsec,
        "scale_arcsec_per_mm": ARCSEC_PER_RAD / config.focal_length_mm,
        "frequency_limit_cycles_per_mm": frequency_limit,
    }


def _image_figure(images: list[tuple[str, np.ndarray]], extent: tuple[float, float, float, float] | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(images), figsize=(4.0 * len(images), 4.1), squeeze=False, constrained_layout=True)
    for axis, (title, image) in zip(axes[0], images):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", extent=extent)
        axis.set_title(title)
        axis.set_xlabel("Angle (arcsec)" if extent else "Pixel")
        axis.set_ylabel("Angle (arcsec)" if extent else "Pixel")
    return fig


def _array_mtf_figure(
    centers_mm: np.ndarray,
    aperture_diameter_mm: float,
    mtf: np.ndarray,
    frequency_limit_cycles_per_mm: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    radius = aperture_diameter_mm / 2.0
    for index, (x_value, y_value) in enumerate(centers_mm, start=1):
        aperture = plt.Circle(
            (x_value, y_value), radius, facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25
        )
        axes[0].add_patch(aperture)
        axes[0].plot(x_value, y_value, "+", color="tab:blue", markersize=8)
        axes[0].annotate(f"P{index}", (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    x_min = float(np.min(centers_mm[:, 0]) - radius)
    x_max = float(np.max(centers_mm[:, 0]) + radius)
    y_min = float(np.min(centers_mm[:, 1]) - radius)
    y_max = float(np.max(centers_mm[:, 1]) + radius)
    span = max(x_max - x_min, y_max - y_min, aperture_diameter_mm)
    padding = max(0.08 * span, 1.0)
    axes[0].set_xlim(x_min - padding, x_max + padding)
    axes[0].set_ylim(y_min - padding, y_max + padding)
    axes[0].set_title("Sub-aperture layout")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.2)
    frequency_extent = (
        -frequency_limit_cycles_per_mm,
        frequency_limit_cycles_per_mm,
        -frequency_limit_cycles_per_mm,
        frequency_limit_cycles_per_mm,
    )
    image = axes[1].imshow(
        mtf, cmap="viridis", origin="lower", vmin=0.0, vmax=1.0, extent=frequency_extent
    )
    axes[1].set_title("Modulation transfer function")
    axes[1].set_xlabel("Spatial frequency X (cycles/mm)")
    axes[1].set_ylabel("Spatial frequency Y (cycles/mm)")
    fig.colorbar(image, ax=axes[1], label="MTF")
    return fig


def _composite_mtf_figure(
    mtf: np.ndarray, frequency_limit_cycles_per_mm: float
) -> plt.Figure:
    limit = frequency_limit_cycles_per_mm
    fig, axis = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    image = axis.imshow(
        mtf,
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        extent=(-limit, limit, -limit, limit),
    )
    axis.set_title("Composite modulation transfer function")
    axis.set_xlabel("Spatial frequency X (cycles/mm)")
    axis.set_ylabel("Spatial frequency Y (cycles/mm)")
    fig.colorbar(image, ax=axis, label="MTF")
    return fig


def _initialize_apertures() -> None:
    if "apertures" not in st.session_state:
        st.session_state.apertures = [
            {"id": 1, "x": "-116.25", "y": "-67.10"},
            {"id": 2, "x": "116.25", "y": "-67.10"},
            {"id": 3, "x": "-116.25", "y": "67.10"},
            {"id": 4, "x": "116.25", "y": "67.10"},
        ]
        st.session_state.next_aperture_id = 5


def _add_aperture() -> None:
    aperture_id = st.session_state.next_aperture_id
    st.session_state.apertures.append({"id": aperture_id, "x": "0.0", "y": "0.0"})
    st.session_state.next_aperture_id += 1


def _remove_aperture(aperture_id: int) -> None:
    if len(st.session_state.apertures) > 1:
        st.session_state.apertures = [
            aperture for aperture in st.session_state.apertures if aperture["id"] != aperture_id
        ]


def _rectangle_apertures(diagonal_mm: float, first_id: int) -> list[dict[str, str | int]]:
    half_angle = np.deg2rad(30.0)
    half_x = diagonal_mm * np.cos(half_angle) / 2.0
    half_y = diagonal_mm * np.sin(half_angle) / 2.0
    return [
        {"id": first_id, "x": f"{-half_x:.6f}", "y": f"{-half_y:.6f}"},
        {"id": first_id + 1, "x": f"{half_x:.6f}", "y": f"{-half_y:.6f}"},
        {"id": first_id + 2, "x": f"{-half_x:.6f}", "y": f"{half_y:.6f}"},
        {"id": first_id + 3, "x": f"{half_x:.6f}", "y": f"{half_y:.6f}"},
    ]


def _default_pose_plan() -> str:
    lines = ["# name array rotation_deg weight noise_sigma seed enabled"]
    for name in ("B1", "B2"):
        for angle in (0.0, 45.0, 90.0, 135.0):
            lines.append(f"{name}_{angle:g} {name} {angle:g} 1 - - 1")
    for index in range(16):
        angle = 11.25 * index
        pose_name = f"B3_{angle:g}".replace(".", "p")
        lines.append(f"{pose_name} B3 {angle:g} 1 - - 1")
    return "\n".join(lines)


def _initialize_multi_state() -> None:
    if "multi_array_configs" not in st.session_state:
        st.session_state.multi_array_configs = {
            "B1": _rectangle_apertures(310.0, 1),
            "B2": _rectangle_apertures(745.556, 5),
            "B3": _rectangle_apertures(800.0, 9),
        }
        st.session_state.multi_next_aperture_id = 13
        st.session_state.multi_selected_array = "B1"
        st.session_state.multi_rename_config_name = "B1"
    if "pose_plan_text" not in st.session_state:
        st.session_state.pose_plan_text = _default_pose_plan()


def _add_multi_aperture(array_name: str) -> None:
    aperture_id = st.session_state.multi_next_aperture_id
    st.session_state.multi_array_configs[array_name].append(
        {"id": aperture_id, "x": "0.0", "y": "0.0"}
    )
    st.session_state.multi_next_aperture_id += 1


def _remove_multi_aperture(array_name: str, aperture_id: int) -> None:
    apertures = st.session_state.multi_array_configs[array_name]
    if len(apertures) > 1:
        st.session_state.multi_array_configs[array_name] = [
            aperture for aperture in apertures if aperture["id"] != aperture_id
        ]


def _input_form() -> PoseInput | None:
    _initialize_apertures()
    defaults = {
        "diameter": "80",
        "focal": "28000",
        "detector": "1.664",
        "noise": "0.02",
        "regularization": "0.002",
        "seed": "7",
        "spectrum": "450 0.2\n500 0.2\n550 0.2\n600 0.2\n650 0.2",
    }
    st.subheader("子孔径构型")
    header_left, header_right = st.columns([5, 1])
    header_left.caption("单位为 mm")
    header_right.button("增加子孔径", on_click=_add_aperture, width="stretch")
    values: dict[str, str] = {}
    aperture_ids: list[int] = []
    for display_index, aperture in enumerate(st.session_state.apertures, start=1):
        aperture_id = aperture["id"]
        aperture_ids.append(aperture_id)
        label_col, x_col, y_col, action_col = st.columns([1.1, 2, 2, 0.8])
        label_col.markdown(f"**P{display_index}**")
        values[f"x{aperture_id}"] = x_col.text_input(
            f"P{display_index} 中心 X", aperture["x"], key=f"aperture_{aperture_id}_x"
        )
        values[f"y{aperture_id}"] = y_col.text_input(
            f"P{display_index} 中心 Y", aperture["y"], key=f"aperture_{aperture_id}_y"
        )
        action_col.button(
            "删除",
            key=f"remove_aperture_{aperture_id}",
            on_click=_remove_aperture,
            args=(aperture_id,),
            disabled=len(st.session_state.apertures) == 1,
            width="stretch",
        )

    st.subheader("其他设置")
    left, right = st.columns(2)
    with left:
        values["diameter"] = st.text_input("统一子孔径口径 (mm)", defaults["diameter"])
        values["focal"] = st.text_input("有效焦距 (mm)", defaults["focal"])
        values["detector"] = st.text_input("探测器边长 (mm)", defaults["detector"])
        values["spectrum"] = st.text_area(
            "光谱 (wavelength nm, relative intensity)",
            defaults["spectrum"],
            height=140,
            help="每行输入：波长(nm) 相对强度；强度会再次归一化。",
        )
    with right:
        values["noise"] = st.text_input("高斯噪声相对强度", defaults["noise"])
        values["regularization"] = st.text_input("Wiener 正则化系数", defaults["regularization"])
        values["seed"] = st.text_input(
            "噪声随机种子（可选）", defaults["seed"], help="留空时每次运行使用新的随机噪声。"
        )
    submitted = st.button("运行单姿态成像", type="primary", width="stretch")
    if not submitted:
        return None
    try:
        return make_pose_input(values, aperture_ids)
    except ValueError as error:
        st.error(str(error))
        return None


def _delete_multi_config(array_name: str) -> None:
    configs = st.session_state.multi_array_configs
    if len(configs) > 1 and array_name in configs:
        del configs[array_name]
        st.session_state.multi_selected_array = next(iter(configs))
        st.session_state.multi_rename_config_name = st.session_state.multi_selected_array


def _sync_multi_rename_name() -> None:
    st.session_state.multi_rename_config_name = st.session_state.multi_selected_array


def _rename_multi_config(array_name: str) -> None:
    new_name = st.session_state.multi_rename_config_name.strip()
    configs = st.session_state.multi_array_configs
    if not new_name or any(char.isspace() for char in new_name):
        st.session_state.multi_config_error = "构型名称不能为空或包含空格"
        return
    if new_name != array_name and new_name in configs:
        st.session_state.multi_config_error = f"构型 {new_name} 已存在"
        return
    if new_name == array_name:
        st.session_state.multi_config_error = ""
        return
    apertures = configs[array_name]
    for aperture in apertures:
        aperture_id = aperture["id"]
        aperture["x"] = st.session_state.get(
            f"multi_{array_name}_{aperture_id}_x", aperture["x"]
        )
        aperture["y"] = st.session_state.get(
            f"multi_{array_name}_{aperture_id}_y", aperture["y"]
        )
    renamed: dict[str, list[dict[str, str | int]]] = {}
    for name, value in configs.items():
        renamed[new_name if name == array_name else name] = value
    st.session_state.multi_array_configs = renamed
    st.session_state.multi_selected_array = new_name
    if st.session_state.get("generator_array") == array_name:
        st.session_state.generator_array = new_name
    updated_lines: list[str] = []
    for line in st.session_state.pose_plan_text.splitlines():
        content, marker, comment = line.partition("#")
        fields = content.split()
        if len(fields) >= 2 and fields[1] == array_name:
            fields[1] = new_name
            content = " ".join(fields)
        updated_lines.append(content + (marker + comment if marker else ""))
    st.session_state.pose_plan_text = "\n".join(updated_lines)
    st.session_state.multi_rename_config_name = new_name
    st.session_state.multi_config_error = ""


def _generate_pose_lines() -> None:
    try:
        array_name = st.session_state.generator_array
        start = _parse_float(st.session_state.generator_start, "初始角度")
        step = _parse_float(st.session_state.generator_step, "角度步长")
        count = int(st.session_state.generator_count.strip())
        if count <= 0:
            raise ValueError
        prefix = st.session_state.generator_prefix.strip()
        if not prefix or any(char.isspace() for char in prefix):
            raise ValueError("名称前缀不能为空或包含空格")
        generated = []
        for index in range(count):
            angle = start + step * index
            name = f"{prefix}_{angle:g}".replace(".", "p")
            generated.append(f"{name} {array_name} {angle:g} 1 - - 1")
        current = st.session_state.pose_plan_text.rstrip()
        st.session_state.pose_plan_text = current + ("\n" if current else "") + "\n".join(generated)
        st.session_state.generator_message = f"已追加 {count} 个姿态。"
        st.session_state.generator_error = ""
    except ValueError as error:
        message = str(error) or "姿态数量必须是正整数"
        st.session_state.generator_error = message
        st.session_state.generator_message = ""


def _multi_pose_input_panel() -> tuple[str | None, MultiPoseInput | None]:
    _initialize_multi_state()
    configs = st.session_state.multi_array_configs
    st.subheader("阵列构型")
    create_name_col, create_button_col = st.columns([4, 1])
    new_name = create_name_col.text_input("新构型名称", key="multi_new_config_name")
    if create_button_col.button("增加构型", width="stretch"):
        normalized_name = new_name.strip()
        if not normalized_name or any(char.isspace() for char in normalized_name):
            st.error("构型名称不能为空或包含空格")
        elif normalized_name in configs:
            st.error(f"构型 {normalized_name} 已存在")
        else:
            aperture_id = st.session_state.multi_next_aperture_id
            configs[normalized_name] = [{"id": aperture_id, "x": "0.0", "y": "0.0"}]
            st.session_state.multi_next_aperture_id += 1
            st.session_state.multi_selected_array = normalized_name
            st.session_state.multi_rename_config_name = normalized_name

    selected_col, add_col, delete_col = st.columns([4, 1, 1])
    selected_name = selected_col.selectbox(
        "当前构型",
        list(configs),
        key="multi_selected_array",
        on_change=_sync_multi_rename_name,
    )
    add_col.button(
        "增加孔径",
        on_click=_add_multi_aperture,
        args=(selected_name,),
        width="stretch",
    )
    delete_col.button(
        "删除构型",
        on_click=_delete_multi_config,
        args=(selected_name,),
        disabled=len(configs) == 1,
        width="stretch",
    )
    rename_name_col, rename_button_col = st.columns([4, 1])
    rename_name_col.text_input("构型新名称", key="multi_rename_config_name")
    rename_button_col.button(
        "重命名",
        on_click=_rename_multi_config,
        args=(selected_name,),
        width="stretch",
    )
    if st.session_state.get("multi_config_error"):
        st.error(st.session_state.multi_config_error)

    values: dict[str, str] = {}
    apertures = configs[selected_name]
    for display_index, aperture in enumerate(apertures, start=1):
        aperture_id = int(aperture["id"])
        key_x = f"multi_{selected_name}_{aperture_id}_x"
        key_y = f"multi_{selected_name}_{aperture_id}_y"
        label_col, x_col, y_col, action_col = st.columns([1.1, 2, 2, 0.8])
        label_col.markdown(f"**P{display_index}**")
        values[key_x] = x_col.text_input(
            f"{selected_name} P{display_index} 中心 X", str(aperture["x"]), key=key_x
        )
        values[key_y] = y_col.text_input(
            f"{selected_name} P{display_index} 中心 Y", str(aperture["y"]), key=key_y
        )
        action_col.button(
            "删除",
            key=f"remove_multi_{selected_name}_{aperture_id}",
            on_click=_remove_multi_aperture,
            args=(selected_name, aperture_id),
            disabled=len(apertures) == 1,
            width="stretch",
        )

    st.subheader("公共光学参数")
    optical_col, noise_col = st.columns(2)
    with optical_col:
        values["multi_diameter"] = st.text_input(
            "统一子孔径口径 (mm)", "80", key="multi_diameter"
        )
        values["multi_focal"] = st.text_input(
            "有效焦距 (mm)", "28000", key="multi_focal"
        )
        values["multi_detector"] = st.text_input(
            "探测器边长 (mm)", "1.664", key="multi_detector"
        )
        values["multi_spectrum"] = st.text_area(
            "光谱 (wavelength nm, relative intensity)",
            "450 0.2\n500 0.2\n550 0.2\n600 0.2\n650 0.2",
            key="multi_spectrum",
            height=140,
        )
    with noise_col:
        values["multi_noise"] = st.text_input(
            "高斯噪声相对强度", "0.02", key="multi_noise"
        )
        values["multi_base_seed"] = st.text_input(
            "基础随机种子（可选）", "1000", key="multi_base_seed"
        )
        values["multi_regularization"] = st.text_input(
            "正则化系数", "0.002", key="multi_regularization"
        )
        values["multi_mtf_threshold"] = st.text_input(
            "MTF 阈值", "0.0", key="multi_mtf_threshold"
        )
        values["multi_transition_width"] = st.text_input(
            "软过渡宽度", "0.04", key="multi_transition_width"
        )

    st.subheader("规则姿态生成")
    generator_cols = st.columns(5)
    if st.session_state.get("generator_array") not in configs:
        st.session_state.generator_array = next(iter(configs))
    generator_cols[0].selectbox("阵列构型", list(configs), key="generator_array")
    generator_cols[1].text_input("起始角度", "0", key="generator_start")
    generator_cols[2].text_input("角度步长", "11.25", key="generator_step")
    generator_cols[3].text_input("姿态数量", "16", key="generator_count")
    generator_cols[4].text_input("名称前缀", "pose", key="generator_prefix")
    st.button("追加到姿态计划", on_click=_generate_pose_lines, width="stretch")
    if st.session_state.get("generator_error"):
        st.error(st.session_state.generator_error)
    elif st.session_state.get("generator_message"):
        st.success(st.session_state.generator_message)

    st.subheader("姿态计划")
    values["pose_plan"] = st.text_area(
        "name array rotation_deg weight noise_sigma seed enabled",
        key="pose_plan_text",
        height=300,
        help="noise_sigma 和 seed 使用 - 时继承公共参数；enabled 使用 1 或 0。",
    )
    method_col, weight_col = st.columns(2)
    values["fusion_method"] = method_col.selectbox(
        "融合方法",
        ["standard", "soft_adaptive"],
        format_func=lambda value: {
            "standard": "标准多姿态 Wiener",
            "soft_adaptive": "软自适应 Wiener",
        }[value],
    )
    values["weight_mode"] = weight_col.selectbox(
        "权重方式",
        ["equal", "inverse_noise", "manual"],
        format_func=lambda value: {
            "equal": "等权",
            "inverse_noise": "噪声方差倒数",
            "manual": "姿态计划手动权重",
        }[value],
    )
    validate_col, run_col = st.columns(2)
    validate = validate_col.button("校验姿态计划", width="stretch")
    run = run_col.button("运行多姿态合成", type="primary", width="stretch")
    if not validate and not run:
        return None, None
    try:
        config = make_multi_pose_input(values, configs)
    except ValueError as error:
        st.error(str(error))
        return None, None
    return ("validate" if validate else "run"), config


def _render_single_pose() -> None:
    config = _input_form()
    if config is None:
        st.info("填写参数后点击“运行单姿态成像”。")
        return
    with st.spinner("正在计算 OTF、退化图像和复原结果…"):
        result = run_simulation(config)
    extent = (-result["fov_arcsec"] / 2, result["fov_arcsec"] / 2, -result["fov_arcsec"] / 2, result["fov_arcsec"] / 2)
    st.success("仿真完成")
    metric_cols = st.columns(4)
    metric_cols[0].metric("几何视场", f"{result['fov_arcsec']:.4f} arcsec")
    metric_cols[1].metric("底片比例尺", f"{result['scale_arcsec_per_mm']:.4f} arcsec/mm")
    metric_cols[2].metric("复原 PSNR", f"{psnr(result['target'], result['reconstruction']):.3f} dB")
    metric_cols[3].metric("复原 SSIM", f"{ssim(result['target'], result['reconstruction']):.4f}")

    st.subheader("阵列与频域")
    st.pyplot(
        _array_mtf_figure(
            result["array_centers_mm"],
            config.aperture_diameter_mm,
            result["mtf"],
            result["frequency_limit_cycles_per_mm"],
        ),
        clear_figure=True,
        width="stretch",
    )
    st.subheader("成像链路")
    st.pyplot(
        _image_figure(
            [
                ("Normalized resolution target", result["target"]),
                ("OTF-degraded image", result["degraded"]),
                ("Noisy observation", result["noisy"]),
                ("Wiener reconstruction", result["reconstruction"]),
            ],
            extent,
        ),
        clear_figure=True,
        width="stretch",
    )
    st.subheader("复原评价")
    metrics = {
        "PSNR (dB)": psnr(result["target"], result["reconstruction"]),
        "Foreground PSNR (dB)": foreground_balanced_psnr(result["target"], result["reconstruction"]),
        "SSIM": ssim(result["target"], result["reconstruction"]),
        "Gradient similarity": gradient_similarity(result["target"], result["reconstruction"]),
    }
    evaluation_cols = st.columns(4)
    for column, (label, value) in zip(evaluation_cols, metrics.items()):
        column.metric(label, f"{value:.6f}")


def _render_multi_pose() -> None:
    action, config = _multi_pose_input_panel()
    if config is None:
        st.info("校验姿态计划或直接运行多姿态合成。")
        return
    if action == "validate":
        weights = _pose_weights(config)
        st.success(
            f"校验通过：{len(config.arrays)} 个阵列构型，{len(config.poses)} 个启用姿态。"
        )
        normalized = weights / float(np.sum(weights))
        summary_lines = [
            f"{pose.name}  array={pose.array_name}  angle={pose.rotation_deg:g} deg  "
            f"weight={weight:.6f}  noise={pose.noise_sigma:g}  seed={pose.seed}"
            for pose, weight in zip(config.poses, normalized)
        ]
        st.code("\n".join(summary_lines), language="text")
        return
    with st.spinner("正在生成各姿态观测并进行频域融合…"):
        result = run_multi_pose_simulation(config)
    extent = (
        -result["fov_arcsec"] / 2,
        result["fov_arcsec"] / 2,
        -result["fov_arcsec"] / 2,
        result["fov_arcsec"] / 2,
    )
    reconstruction = result["reconstruction"]
    target = result["target"]
    st.success(f"已完成 {int(result['pose_count'])} 个姿态的融合。")
    metric_cols = st.columns(4)
    metric_cols[0].metric("姿态数", str(int(result["pose_count"])))
    metric_cols[1].metric("PSNR", f"{psnr(target, reconstruction):.3f} dB")
    metric_cols[2].metric("SSIM", f"{ssim(target, reconstruction):.4f}")
    metric_cols[3].metric(
        "Gradient similarity", f"{gradient_similarity(target, reconstruction):.4f}"
    )
    st.subheader("复合频域覆盖")
    st.pyplot(
        _composite_mtf_figure(
            result["composite_mtf"], result["frequency_limit_cycles_per_mm"]
        ),
        clear_figure=True,
        width="stretch",
    )
    st.subheader("多姿态成像链路")
    first_name = result["pose_names"][0]
    st.pyplot(
        _image_figure(
            [
                ("Normalized resolution target", target),
                (f"First observation: {first_name}", result["first_observation"]),
                ("Multi-pose Wiener reconstruction", reconstruction),
            ],
            extent,
        ),
        clear_figure=True,
        width="stretch",
    )
    secondary_cols = st.columns(3)
    secondary_cols[0].metric("Field of view", f"{result['fov_arcsec']:.4f} arcsec")
    secondary_cols[1].metric(
        "Plate scale", f"{result['scale_arcsec_per_mm']:.4f} arcsec/mm"
    )
    secondary_cols[2].metric(
        "Foreground PSNR", f"{foreground_balanced_psnr(target, reconstruction):.3f} dB"
    )


def main() -> None:
    st.set_page_config(page_title="Fizeau 成像模拟", page_icon="◌", layout="wide")
    st.title("Fizeau 干涉阵成像模拟")
    mode = st.radio("成像模式", ["单姿态", "多姿态合成"], horizontal=True)
    if mode == "单姿态":
        _render_single_pose()
    else:
        _render_multi_pose()


if __name__ == "__main__":
    main()
