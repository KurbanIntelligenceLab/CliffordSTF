"""OmegaConf-based config loader.

Merge chain (later overrides earlier):
    1. ``<pkg>/configs/base.yaml``        — global defaults
    2. ``<pkg>/configs/dataset/{name}.yaml`` — dataset-specific defaults
    3. ``<pkg>/configs/model/{name}.yaml`` — model architecture defaults
    4. ``--config <path.yaml>``           — optional user override file
    5. CLI dot-overrides                   — e.g. ``training.lr=1e-3``

Design notes:

- The packaged ``cliffordstf/configs/`` directory is the default config
  root (resolved relative to this module so editable installs and built
  wheels both work).
- Callers may pass ``extra_search_paths`` so the optional ``baselines``
  package can plug its own ``configs/`` directory into the lookup
  without using a global registry (Amendment 1).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIGS_DIR: Path = Path(__file__).resolve().parent.parent / "configs"
"""Packaged ``cliffordstf/configs/`` directory."""


def _find_yaml(name: str, subdir: str, search_paths: Sequence[Path]) -> Path:
    """Return the first ``<search>/{subdir}/{name}.yaml`` that exists."""
    seen: list[Path] = []
    for root in search_paths:
        candidate = root / subdir / f"{name}.yaml"
        seen.append(candidate)
        if candidate.exists():
            return candidate
    available: list[str] = []
    for root in search_paths:
        sub = root / subdir
        if sub.is_dir():
            available.extend(sorted(p.stem for p in sub.glob("*.yaml")))
    raise FileNotFoundError(
        f"Config file not found for {subdir}/{name}.yaml.\n"
        f"  searched: {[str(p) for p in seen]}\n"
        f"  available: {sorted(set(available))}"
    )


def load_config(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    extra_search_paths: Sequence[Path] | None = None,
) -> DictConfig:
    """Build a merged ``DictConfig`` from packaged YAMLs and CLI argv.

    Args:
        argv: CLI arguments. ``None`` falls back to ``sys.argv[1:]``.
        defaults: Hard-coded defaults merged after ``base.yaml`` and
            before ``dataset.yaml`` (useful for trainers that pre-pin
            ``dataset.name``).
        extra_search_paths: Additional config roots searched after
            ``DEFAULT_CONFIGS_DIR``. Used to plug in ``baselines``-style
            third-party config directories.

    Returns:
        The merged ``DictConfig`` with interpolations resolved.
    """
    if argv is None:
        argv = sys.argv[1:]

    override_yaml_path, dot_overrides = _split_argv(argv)
    cli_conf = OmegaConf.from_dotlist(list(dot_overrides)) if dot_overrides else OmegaConf.create()
    defaults_conf = OmegaConf.create(defaults) if defaults else OmegaConf.create()

    search_paths: list[Path] = [DEFAULT_CONFIGS_DIR]
    if extra_search_paths:
        search_paths.extend(extra_search_paths)

    base_conf = _load_base(search_paths)
    peek = OmegaConf.merge(base_conf, defaults_conf, cli_conf)
    if not isinstance(peek, DictConfig):
        raise TypeError(f"Peek merge produced non-dict config: {type(peek).__name__}")

    dataset_conf = _maybe_load(peek, "dataset.name", "dataset", search_paths)
    model_conf = _maybe_load(peek, "model.name", "model", search_paths)

    override_conf = OmegaConf.load(override_yaml_path) if override_yaml_path else OmegaConf.create()

    cfg = OmegaConf.merge(
        base_conf, defaults_conf, dataset_conf, model_conf, override_conf, cli_conf
    )
    OmegaConf.resolve(cfg)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Merged config is not a DictConfig: {type(cfg).__name__}")
    return cfg


def _split_argv(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Separate ``--config <path>`` from OmegaConf dot-overrides."""
    override_yaml: str | None = None
    dot_overrides: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            override_yaml = argv[i + 1]
            i += 2
        else:
            dot_overrides.append(argv[i])
            i += 1
    return override_yaml, dot_overrides


def _load_base(search_paths: Sequence[Path]) -> DictConfig:
    for root in search_paths:
        candidate = root / "base.yaml"
        if candidate.exists():
            loaded = OmegaConf.load(candidate)
            if isinstance(loaded, DictConfig):
                return loaded
    return OmegaConf.create({})


def _maybe_load(
    peek: DictConfig,
    selector: str,
    subdir: str,
    search_paths: Sequence[Path],
) -> DictConfig:
    """Load ``<subdir>/{<peek.selector>}.yaml`` if the name is set; else empty."""
    name = OmegaConf.select(peek, selector, default=None)
    if not (name and isinstance(name, str)):
        return OmegaConf.create({})
    try:
        path = _find_yaml(name, subdir, search_paths)
    except FileNotFoundError:
        return OmegaConf.create({})
    loaded = OmegaConf.load(path)
    if isinstance(loaded, DictConfig):
        return loaded
    return OmegaConf.create({})


__all__ = ["DEFAULT_CONFIGS_DIR", "load_config"]
