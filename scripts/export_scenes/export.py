"""Export mjlab task scenes (scene.xml + assets/) into ``public/envs/<env_id>/``.

Usage::

  uv run python export.py                   # export all configured envs
  uv run python export.py g1_rough          # export a specific env id
  uv run python export.py --assets-only     # copy assets referenced by existing scene.xml files

The output directory layout matches what the web app expects.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401  (registers tasks)
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_SRC = REPO_ROOT / "mjlab_examples" / "src"
if EXAMPLES_SRC.exists():
  sys.path.insert(0, str(EXAMPLES_SRC))
  import mjlab_examples.tasks  # noqa: F401  (registers example tasks)


@dataclass(frozen=True)
class ExportTarget:
  env_id: str
  task_id: str
  use_play_cfg: bool = True


# Web-side env id -> mjlab task id. ``use_play_cfg`` selects the smaller play
# variant of the env config (5x5 terrain, no domain randomization).
TARGETS: tuple[ExportTarget, ...] = (
  ExportTarget("g1_flat", "Mjlab-Velocity-Flat-Unitree-G1"),
  ExportTarget("g1_rough", "Mjlab-Velocity-Rough-Unitree-G1"),
  ExportTarget("go1_flat", "Mjlab-Velocity-Flat-Unitree-Go1"),
  ExportTarget("go1_rough", "Mjlab-Velocity-Rough-Unitree-Go1"),
  ExportTarget("g1_backflip", "Mjlab-Tracking-Flat-Unitree-G1"),
)


def public_envs_dir() -> Path:
  # scripts/export_scenes/export.py -> repo_root/public/envs
  return Path(__file__).resolve().parents[2] / "public" / "envs"


_FILE_REF_RE = re.compile(r'file="([^"]+)"')


def _build_mesh_index() -> dict[str, Path]:
  """Map basename -> source path for every mesh shipped by mjlab."""
  asset_zoo = Path(mjlab.__file__).resolve().parent / "asset_zoo"
  index: dict[str, Path] = {}
  for path in asset_zoo.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".stl", ".obj", ".png", ".dae"}:
      index.setdefault(path.name, path)
  return index


def _copy_referenced_assets(scene_xml: Path, mesh_index: dict[str, Path]) -> None:
  assets_dir = scene_xml.parent / "assets"
  referenced = sorted(set(_FILE_REF_RE.findall(scene_xml.read_text())))
  if not referenced:
    return
  assets_dir.mkdir(parents=True, exist_ok=True)
  missing: list[str] = []
  for ref in referenced:
    name = Path(ref).name
    source = mesh_index.get(name)
    if source is None:
      missing.append(ref)
      continue
    dest = assets_dir / ref
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
  if missing:
    print(f"  ! missing {len(missing)} asset(s): {missing}")


def export_target(
  target: ExportTarget,
  out_root: Path,
  mesh_index: dict[str, Path],
) -> None:
  print(f"[export] {target.env_id} <- {target.task_id}")
  env_cfg = load_env_cfg(target.task_id, play=target.use_play_cfg)
  scene = Scene(env_cfg.scene, device="cpu")

  out_dir = out_root / target.env_id
  if out_dir.exists():
    shutil.rmtree(out_dir)
  scene.write(out_dir, zip=False)
  _copy_referenced_assets(out_dir / "scene.xml", mesh_index)
  print(f"  -> wrote {out_dir}")


def copy_target_assets(
  target: ExportTarget,
  out_root: Path,
  mesh_index: dict[str, Path],
) -> None:
  scene_xml = out_root / target.env_id / "scene.xml"
  if not scene_xml.exists():
    raise FileNotFoundError(f"Missing scene XML for {target.env_id}: {scene_xml}")
  print(f"[assets] {target.env_id}")
  _copy_referenced_assets(scene_xml, mesh_index)


def main() -> int:
  args = sys.argv[1:]
  assets_only = False
  selected: list[str] = []
  for arg in args:
    if arg == "--assets-only":
      assets_only = True
    else:
      selected.append(arg)

  out_root = public_envs_dir()
  out_root.mkdir(parents=True, exist_ok=True)

  targets = TARGETS
  if selected:
    by_id = {t.env_id: t for t in TARGETS}
    unknown = [s for s in selected if s not in by_id]
    if unknown:
      print(f"Unknown env id(s): {unknown}")
      print(f"Available: {sorted(by_id)}")
      return 2
    targets = tuple(by_id[s] for s in selected)

  mesh_index = _build_mesh_index()
  for target in targets:
    if assets_only:
      copy_target_assets(target, out_root, mesh_index)
    else:
      export_target(target, out_root, mesh_index)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
