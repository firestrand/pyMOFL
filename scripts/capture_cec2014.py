"""Capture pinned official CEC2014 references from a supplied local archive.

Explicit setup only: no network or implicit reference-engine dependency.
Reference source and data rights are unspecified; output remains external.
"""

import argparse
import difflib
import hashlib
import json
import math
import platform
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ARCHIVE = args.archive.resolve()
    OUT = args.output.resolve()
    if OUT.is_relative_to(ROOT):
        parser.error("Capture output must be outside the repository")
    if OUT.exists() and any(OUT.iterdir()):
        parser.error("Capture output directory must be empty")

    EXPECTED = "1a210560398ca7a50be6adf1e5e90602222519ef23b6e31aba8847e109761876"
    WORK = OUT / "engine"

    if _sha256(ARCHIVE.read_bytes()) != EXPECTED:
        raise ValueError("Archive SHA256 does not match the approved source")
    WORK.mkdir(parents=True, exist_ok=True)
    data01 = ROOT / "tests/validation_data/cec/2005/f01.json"
    if (
        _sha256(data01.read_bytes())
        != "5e7268c46c288a287e1909b5986518631ea87f08e16cb2ace0517cc84ef0142a"
    ):
        raise ValueError("DATA-01 inputs no longer match the approved capture")
    inventory = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive member")
            if not entry.is_dir() and name.parts[:2] == ("cec14-c-code", "input_data"):
                target = WORK.joinpath(*name.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(entry)
                target.write_bytes(data)
                inventory.append({"path": str(name), "sha256": _sha256(data)})
        original = archive.read("cec14-c-code/cec14_test_func.cpp")
        text = original.decode("ascii", errors="strict")
        if text.count("#include <WINDOWS.H>") != 1 or text.count("%Lf") != 5:
            raise ValueError("Unexpected official source patch sites")
        patched = text.replace(
            "#include <WINDOWS.H>", "// Unused Windows header removed for Linux."
        ).replace("%Lf", "%lf")
        (WORK / "cec14_test_func.cpp").write_bytes(patched.encode("ascii"))
        patch = "".join(
            difflib.unified_diff(
                text.splitlines(True),
                patched.splitlines(True),
                fromfile="official/cec14_test_func.cpp",
                tofile="linux/cec14_test_func.cpp",
            )
        )
        (WORK / "portability.patch").write_text(patch)

    driver = """#include <iostream>
    #include <iomanip>
    #include <vector>
    #include <cmath>
    void cec14_test_func(double *, double *,int,int,int);
    double *OShift=nullptr,*M=nullptr,*y=nullptr,*z=nullptr,*x_bound=nullptr;
    int ini_flag=0,n_flag=0,func_flag=0,*SS=nullptr;
    int main() {
     int fid, dim, rows;
     if (!(std::cin >> fid >> dim >> rows) || fid<1 || fid>30 ||
         (dim!=10 && dim!=30 && dim!=50) || rows!=5) return 2;
     std::vector<double> X(rows*dim), values(rows);
     for (double& value : X) if (!(std::cin >> value) || !std::isfinite(value)) return 3;
     cec14_test_func(X.data(),values.data(),dim,rows,fid);
     std::cout << std::setprecision(17);
     for (double value : values) { if (!std::isfinite(value)) return 4; std::cout << value << "\\n"; }
     return 0;
    }
    """
    (WORK / "driver.cpp").write_text(driver)
    flags = [
        "-std=c++17",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Wformat=2",
    ]
    compiled = subprocess.run(
        ["g++", *flags, "cec14_test_func.cpp", "driver.cpp", "-o", "capture"],
        cwd=WORK,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    (WORK / "compiler.log").write_text(compiled.stderr)
    data01 = ROOT / "tests/validation_data/cec/2005/f01.json"
    cases = {row["dimension"]: row for row in json.loads(data01.read_text())["cases"]}
    files = []
    for dimension in (10, 30, 50):
        row = cases[dimension]
        for fid in range(1, 31):
            shifts = (WORK / f"input_data/shift_data_{fid}.txt").read_text().splitlines()
            shift_groups = 10 if fid >= 23 else 1
            if len(shifts) < shift_groups or any(
                len(shifts[k].split()) < dimension for k in range(shift_groups)
            ):
                raise ValueError("Incomplete reference shift data")
            if not all(
                math.isfinite(float(v)) for k in range(shift_groups) for v in shifts[k].split()
            ):
                raise ValueError("Nonfinite reference shift data")
            xshift = [float(v) for v in shifts[0].split()][:dimension]
            matrix = WORK / f"input_data/M_{fid}_D{dimension}.txt"
            matrix_values = [float(v) for v in matrix.read_text().split()]
            if len(matrix_values) != dimension * dimension * (10 if fid >= 23 else 1) or not all(
                math.isfinite(v) for v in matrix_values
            ):
                raise ValueError("Invalid reference matrix data")
            if 17 <= fid <= 22 or fid in (29, 30):
                shuffle = [
                    int(v)
                    for v in (WORK / f"input_data/shuffle_data_{fid}_D{dimension}.txt")
                    .read_text()
                    .split()
                ]
                groups = 10 if fid in (29, 30) else 1
                if len(shuffle) != dimension * groups or any(
                    sorted(shuffle[k * dimension : (k + 1) * dimension])
                    != list(range(1, dimension + 1))
                    for k in range(groups)
                ):
                    raise ValueError("Invalid reference shuffle data")
            inputs = [
                ("shift", xshift),
                ("zeros", [0.0] * dimension),
                ("random", row["random_input"]),
                ("bounds_min", row["operational_bounds"]["low"]),
                ("bounds_max", row["operational_bounds"]["high"]),
            ]
            if not all(len(x) == dimension for _, x in inputs):
                raise ValueError("Input dimensions do not match the capture contract")
            payload = (
                f"{fid} {dimension} 5\n"
                + "\n".join(" ".join(format(v, ".17g") for v in x) for _, x in inputs)
                + "\n"
            )
            command = [str(WORK / "capture")]
            run = subprocess.run(
                command,
                input=payload,
                cwd=WORK,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            repeat = subprocess.run(
                command,
                input=payload,
                cwd=WORK,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            if run.stdout != repeat.stdout:
                raise ValueError("Reference capture was not repeatable")
            values = [float(value) for value in run.stdout.splitlines()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError("Reference engine returned invalid output")
            if fid == 1 and values[0] != 100.0:
                raise ValueError("F1 shift control did not return 100.0")
            records = [
                {"func_id": fid, "dim": dimension, "case": name, "x": x, "value": value}
                for (name, x), value in zip(inputs, values, strict=True)
            ]
            target = OUT / f"datasets/CEC2014/func_{fid}_D{dimension}/golden.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            data = "".join(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
            ).encode()
            target.write_bytes(data)
            files.append(
                {
                    "path": str(target.relative_to(OUT)),
                    "sha256": _sha256(data),
                    "records": len(records),
                    "input_sha256": _sha256(payload.encode()),
                }
            )
    manifest = {
        "source": "https://github.com/P-N-Suganthan/CEC2014",
        "revision": "98488087d590c29aaded9978ccfe2a356d10dd63",
        "archive_sha256": EXPECTED,
        "original_source_sha256": _sha256(original),
        "patched_source_sha256": _sha256(patched.encode("ascii")),
        "portability_patch_sha256": _sha256(patch.encode()),
        "driver_sha256": _sha256(driver.encode()),
        "executable_sha256": _sha256((WORK / "capture").read_bytes()),
        "compiler": subprocess.check_output(
            ["g++", "--version"], text=True, timeout=30
        ).splitlines()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler_flags": flags,
        "data01_sha256": _sha256(data01.read_bytes()),
        "rights": "unspecified; no redistribution of source/data",
        "data_files": inventory,
        "captures": files,
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "captures": len(files),
                "records": sum(f["records"] for f in files),
                "manifest_sha256": _sha256((OUT / "provenance.json").read_bytes()),
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
