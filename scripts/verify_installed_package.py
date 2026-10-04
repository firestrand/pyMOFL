"""Verify core and CLI wheels in clean environments outside the source checkout.

Dependency installation is explicit integration setup, not part of default pytest.
Pass an exported lockfile with --constraints for the locked artifact gate; omit it
to exercise the package's declared compatible dependency ranges.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
CAPTURE = CHECKOUT / "tests/validation_data/cec/2005/f01.json"

SMOKE = """
import importlib.util
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pyMOFL

record = json.load(sys.stdin)
package_path = Path(pyMOFL.__file__).resolve()
assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
assert not package_path.is_relative_to(Path(record['checkout'])), package_path
assert pyMOFL.__version__ == record['version'] == version('pyMOFL')
if record['mode'] == 'core':
    for name in ('typer', 'rich', 'mkdocs', 'pytest'):
        assert importlib.util.find_spec(name) is None, name
    assert 'pyMOFL.cli' not in sys.modules
assert 'pyMOFL.definition' not in sys.modules
assert 'pyMOFL.reference_validation' not in sys.modules
from pyMOFL.definition import export_definition, reconstruct_definition
from pyMOFL.reference_validation import prepare_reference_manifest, validate_reference_manifest
case = record['case']
function = pyMOFL.load('cec2005_f01', dimension=case['dimension'])
np.testing.assert_allclose(function.evaluate(np.asarray(case['optimum'])),
                           case['outputs']['optimum'], rtol=0, atol=1e-12)
X = np.asarray([case['optimum'], case['random_input'],
                case['operational_bounds']['low'], case['operational_bounds']['high']],
               dtype=np.float64)
expected = np.asarray([case['outputs'][key] for key in ('optimum', 'random', 'lower', 'upper')])
out = np.empty(len(X), dtype=np.float64)
assert pyMOFL.evaluate_chunks(function, X, 3, deterministic=True,
                              batch_independent=True, out=out) is out
np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-8)
definition = export_definition('cec2005_f01', suite='cec2005', dimension=case['dimension'])
replayed = reconstruct_definition(json.loads(json.dumps(definition, allow_nan=False)))
np.testing.assert_array_equal(replayed.evaluate_batch(X), function.evaluate_batch(X))
bbob_definition = export_definition('bbob_f01', suite='bbob', dimension=2)
bbob_replayed = reconstruct_definition(json.loads(json.dumps(bbob_definition, allow_nan=False)))
np.testing.assert_array_equal(bbob_replayed.evaluate_batch(X[:, :2]),
                            pyMOFL.load('bbob_f01', suite='bbob', dimension=2).evaluate_batch(X[:, :2]))
capture_root = os.environ.get('PYMOFL_REFERENCE_CAPTURE_ROOT')
if capture_root:
    manifest = prepare_reference_manifest(capture_root)
    validation = validate_reference_manifest(json.loads(json.dumps(manifest, allow_nan=False)),
                                             capture_root=capture_root)
    assert validation['success'] is True
    assert validation['scalar_executed'] == validation['batch_executed'] == 450
    if record['mode'] == 'cli':
        with Path(record['manifest_path']).open('x') as destination:
            destination.write(json.dumps(manifest, allow_nan=False))
for year in (2007, 2011):
    suite = pyMOFL.get_suite(f'spso{year}')
    assert [entry.dimension for entry in suite] == [2, 42, 4, 3]
    tripod = pyMOFL.load(f'spso{year}_f04')
    point, value = tripod.base_function.get_global_minimum()
    np.testing.assert_allclose(tripod.evaluate(point), value, rtol=0, atol=1e-12)
assert type(pyMOFL.load('sphere', dimension=case['dimension'])) is \
       type(pyMOFL.load('SphereFunction', dimension=case['dimension']))
print(json.dumps({'mode': record['mode'], 'version': version('pyMOFL'),
                  'python': sys.version.split()[0], 'numpy': version('numpy'),
                  'matplotlib': version('matplotlib')}))
"""


def run(arguments: list[str], *, cwd: Path, input_text: str | None = None) -> str:
    """Run bounded subprocesses without inherited source import overrides."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=180,
        check=True,
    )
    return result.stdout.strip()


def verify(wheel: Path, constraints: Path | None = None) -> list[dict[str, str]]:
    """Install and inspect the actual wheel in core-only and CLI environments."""
    wheel = wheel.resolve(strict=True)
    if wheel.is_dir():
        wheels = list(wheel.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(
                "Wheel directory must contain exactly one wheel; otherwise select a file"
            )
        wheel = wheels[0]
    if constraints is not None:
        constraints = constraints.resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith("/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
    if metadata["Name"].lower() != "pymofl":
        raise ValueError("Expected a pyMOFL wheel")
    version = metadata["Version"]
    case = next(
        case for case in json.loads(CAPTURE.read_text())["cases"] if case["dimension"] == 10
    )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be installed to provision artifact environments")
    reports = []
    with tempfile.TemporaryDirectory(prefix="pymofl-artifact-") as temporary:
        workspace = Path(temporary)
        for mode in ("core", "cli"):
            environment = workspace / mode
            run([uv, "venv", "--python", sys.executable, str(environment)], cwd=workspace)
            executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            specification = str(wheel) + ("[cli]" if mode == "cli" else "")
            install = [uv, "pip", "install", "--python", str(executable), specification]
            if constraints is not None:
                install.extend(["--constraints", str(constraints)])
            run(install, cwd=workspace)
            payload = json.dumps(
                {
                    "checkout": str(CHECKOUT),
                    "version": version,
                    "mode": mode,
                    "case": case,
                    "manifest_path": str(workspace / "reference-manifest.json"),
                }
            )
            reports.append(
                json.loads(run([str(executable), "-c", SMOKE], cwd=workspace, input_text=payload))
            )
            if mode == "cli":
                entry = environment / ("Scripts/pymofl.exe" if os.name == "nt" else "bin/pymofl")
                if run([str(entry), "--version"], cwd=workspace) != version:
                    raise RuntimeError("Installed CLI reported an incorrect package version")
                run([str(entry), "--help"], cwd=workspace)
                capture_root = os.environ.get("PYMOFL_REFERENCE_CAPTURE_ROOT")
                if capture_root:
                    report_path = workspace / "reference-report.json"
                    run(
                        [
                            str(entry),
                            "validate",
                            "--capture-root",
                            capture_root,
                            "--manifest",
                            str(workspace / "reference-manifest.json"),
                            "--report",
                            str(report_path),
                        ],
                        cwd=workspace,
                    )
                    validation = json.loads(report_path.read_text())
                    if not validation["success"] or (
                        validation["scalar_executed"],
                        validation["batch_executed"],
                    ) != (450, 450):
                        raise RuntimeError("Installed CLI did not execute every required case")
    return reports


def main() -> None:
    """Verify a wheel and print observed environment versions as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", required=True, type=Path, help="Wheel file or directory containing one wheel"
    )
    parser.add_argument("--constraints", type=Path)
    arguments = parser.parse_args()
    try:
        reports = verify(arguments.wheel, arguments.constraints)
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode) from error
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
