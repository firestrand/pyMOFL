"""Script to generate docs/function_catalog.md from registered pyMOFL components."""

import inspect
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import _COMPONENTS, scan_package

scan_package()


def _heading_slug(value: str) -> str:
    """Match Python-Markdown's default heading slug, which MkDocs uses for anchors."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s-]", "", normalized).strip().lower()
    return re.sub(r"[-\s]+", "-", cleaned)


class_to_aliases = defaultdict(list)
for alias, cls in _COMPONENTS.items():
    if isinstance(cls, type) and issubclass(cls, OptimizationFunction):
        class_to_aliases[cls].append(alias)


def categorize(cls_name: str, mod: str, default_dim: int | None) -> str:
    if "mishra" in mod or "mishra" in cls_name.lower():
        return "Mishra Family"
    if "schwefel" in mod or "schwefel" in cls_name.lower():
        return "Schwefel Family"
    if cls_name in {
        "GearTrainFunction",
        "CompressionSpringFunction",
        "NetworkFunction",
        "TripodFunction",
        "LennardJonesFunction",
        "ColaFunction",
    }:
        return "Engineering & Special Benchmarks"
    if cls_name in {
        "LinearSlopeFunction",
        "AttractiveSectorFunction",
        "SharpRidgeFunction",
        "BucheRastriginFunction",
        "StepEllipsoidFunction",
        "GallagherPeaksFunction",
        "DifferentPowersFunction",
    }:
        return "BBOB Primitives"
    if default_dim == 2:
        return "Fixed 2D Functions"
    elif default_dim in {3, 4, 5, 6}:
        return "Fixed Dimension (3D-6D) Functions"
    else:
        return "Scalable Functions"


categories = defaultdict(list)
for cls, aliases in class_to_aliases.items():
    mod = cls.__module__
    doc = (cls.__doc__ or "").strip().split("\n")[0]
    doc = doc.replace("|", "\\|")
    sig = inspect.signature(cls.__init__)
    dim_param = sig.parameters.get("dimension")
    default_dim = (
        dim_param.default if dim_param and dim_param.default != inspect.Parameter.empty else None
    )

    file_path = inspect.getfile(cls)
    abs_path = str(Path(file_path).resolve())
    rel_path = str(Path(file_path).relative_to(Path.cwd()))

    cat = categorize(cls.__name__, mod, default_dim)
    categories[cat].append(
        {
            "name": cls.__name__,
            "aliases": sorted(aliases),
            "dim": f"D={default_dim}" if default_dim else "Scalable (D ≥ 1)",
            "abs_path": abs_path,
            "rel_path": rel_path,
            "doc": doc,
        }
    )

order = [
    "Scalable Functions",
    "BBOB Primitives",
    "Fixed 2D Functions",
    "Fixed Dimension (3D-6D) Functions",
    "Mishra Family",
    "Schwefel Family",
    "Engineering & Special Benchmarks",
]

lines = [
    "# pyMOFL Benchmark Function Catalog",
    "",
    (
        f"This catalog provides a comprehensive index of all **{len(class_to_aliases)} concrete"
        f" benchmark function classes** and **{len(_COMPONENTS)} registered component aliases** in"
        " `pyMOFL`."
    ),
    "",
    "## Quick Usage",
    "",
    "Functions can be instantiated either by direct class import or dynamically via the component registry:",
    "",
    "```python",
    "# 1. Direct class import",
    "from pyMOFL.functions.benchmark import SphereFunction, RastriginFunction",
    "f1 = SphereFunction(dimension=10)",
    "",
    "# 2. Dynamic lookup via registry alias",
    "from pyMOFL.registry import get",
    'SphereCls = get("sphere")',
    "f2 = SphereCls(dimension=10)",
    "```",
    "",
    "## Categories",
    "",
]

for cat in order:
    slug = _heading_slug(cat)
    count = len(categories[cat])
    lines.append(f"- [{cat} ({count} functions)](#{slug})")

lines.append("")
lines.append("---")
lines.append("")

for cat in order:
    items = sorted(categories[cat], key=lambda x: x["name"])
    lines.append(f"## {cat}")
    lines.append("")
    lines.append("| Function Class | Registry Aliases | Dimension | Source File | Description |")
    lines.append("|:---|:---|:---:|:---|:---|")
    for it in items:
        aliases_str = ", ".join(f"`{a}`" for a in it["aliases"])
        file_link = f"[{Path(it['rel_path']).name}](../{it['rel_path']})"
        lines.append(
            f"| `{it['name']}` | {aliases_str} | {it['dim']} | {file_link} | {it['doc']} |"
        )
    lines.append("")

output_path = Path("docs/function_catalog.md")
output_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote catalog with {len(class_to_aliases)} classes to {output_path}")
