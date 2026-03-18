from dataclasses import dataclass, field

from src.constants import FieldName, Names, OutputsType
from src.utils.typing import Any, Dict, List, Optional, Set

from .base import BaseMaterial


class MultipleImportanceMonteCarloEnvironmentShader(BaseMaterial):
    """Inference-only export material.

    The original Monte-Carlo shader path is intentionally removed in this
    release build. ReLi3D mesh export only needs material map extraction.
    """

    @dataclass
    class Config(BaseMaterial.Config):
        # Keep legacy fields for backwards-compatible config loading.
        detached_sampling: bool = True
        sampling_stategies: List[Dict[str, Any]] = field(default_factory=list)

        sampler: str = "uniform"
        sample_rotation: bool = False
        sample_rotation_scale: float = 0.025

        illumination_representation: str = "spherical"

        use_power_heuristic: bool = False
        radiance_clamping_upper_limit: Optional[float] = None

        visibility_tester_cls: Optional[str] = None
        visibility_tester: dict = field(default_factory=dict)
        visibility_fade_steps: int = 0

        is_basecolor_metallic: bool = True
        base_reflectivity: float = 0.04

        perceptual_roughness: bool = True
        ndf: str = "ggx"
        geo_shadowing: str = "smith_ue4schlick_ggx"
        fresnel: str = "schlick"

    cfg: Config

    def configure(self) -> None:
        # Keep tone mapping module construction so checkpoint keys under
        # material.tone_mapping.* remain load-compatible.
        super().configure()

    @property
    def requires_full_images(self):
        return False

    def consumed_keys(self) -> Set[FieldName]:
        keys: Set[FieldName] = {
            Names.ROUGHNESS,
            Names.GEOMETRY_NORMAL,
            Names.SHADING_NORMAL,
        }
        if self.cfg.is_basecolor_metallic:
            keys |= {Names.BASECOLOR, Names.METALLIC}
        else:
            keys |= {Names.DIFFUSE, Names.SPECULAR}
        return keys

    def produced_keys(self) -> Set[FieldName]:
        keys: Set[FieldName] = {
            Names.ROUGHNESS,
            Names.SHADING_NORMAL,
        }
        if self.cfg.is_basecolor_metallic:
            keys |= {Names.BASECOLOR, Names.METALLIC}
        else:
            keys |= {Names.DIFFUSE, Names.SPECULAR}
        return keys

    def forward_impl(self, outputs: OutputsType, shade_ray_samples: bool = False) -> OutputsType:
        raise RuntimeError(
            "Shader forward path is not part of this inference-only build. "
            "Use `export(...)` to extract material maps."
        )

    def export_impl(self, outputs: OutputsType) -> OutputsType:
        base = {Names.ROUGHNESS: outputs[Names.ROUGHNESS]}

        if self.cfg.is_basecolor_metallic:
            base[Names.BASECOLOR] = outputs[Names.BASECOLOR]
            base[Names.METALLIC] = outputs[Names.METALLIC]
        else:
            base[Names.DIFFUSE] = outputs[Names.DIFFUSE]
            base[Names.SPECULAR] = outputs[Names.SPECULAR]

        return base
