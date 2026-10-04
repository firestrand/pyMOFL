"""Explicit authorized SPSO boundary inputs; expected values come from C sources."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from capture_spso import CHECKOUT, capture, prepare

BOUNDARY_MAIN = r"""
int main(void) {
  E=exp((long double)1); pi=acos((long double)-1);
  int fid,dim,count;
  if(scanf("%d %d %d",&fid,&dim,&count)!=3 || count<1 || count>4096) return 2;
  if(fid!=4 && fid!=11 && fid!=18 && fid!=21) return 2;
  struct problem pb=problemDef(fid); NATIVE_ADAPTER
  if(dim!=pb.SS.D) return 2;
  for(int i=0;i<count;i++) {
    struct position x={0}; x.size=dim;
    for(int d=0;d<dim;d++) if(scanf("%lf",&x.x[d])!=1 || !isfinite(x.x[d])) return 3;
    struct position q=quantis(x,pb.SS);
    double value=perf(q,fid,pb.SS,pb.objective);
    if(!isfinite(value)) return 4;
    printf("{\"func_id\":%d,\"case\":%d,\"x\":",fid,i);
    array_json(x.x,dim); printf(",\"quantized_x\":"); array_json(q.x,dim);
    printf(",\"value\":%.17g}\n",value);
  }
  return 0;
}
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def boundary_inputs(metadata: dict) -> list[list[float]]:
    low, high, steps = metadata["low"], metadata["high"], metadata["steps"]
    if metadata["func_id"] == 4:
        return [[x, y] for x in (low[0], 0.0, high[0]) for y in (low[1], 0.0, high[1])]
    points = []
    for coordinate, step in enumerate(steps):
        if step <= 1e-40:
            continue
        for bound, direction in ((low[coordinate], 0.5), (high[coordinate], -0.5)):
            tie = (math.floor(bound / step) + direction) * step
            for value in (np.nextafter(tie, -np.inf), tie, np.nextafter(tie, np.inf)):
                point = list(low)
                point[coordinate] = float(value)
                points.append(point)
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(CHECKOUT) or (output.exists() and any(output.iterdir())):
        parser.error("Use an empty external output directory")
    prepare(args.archives.resolve(), output)
    capture(output)  # Retain the original supported source controls unchanged.
    records = json.loads((output / "provenance.json").read_text())
    for record in records:
        root = output / str(record["year"])
        original_driver = (root / "driver.c").read_text()
        driver = original_driver[: original_driver.index("int main(void)")] + BOUNDARY_MAIN.replace(
            "NATIVE_ADAPTER", "pb.SS.normalise=0;" if record["year"] == 2011 else ""
        )
        (root / "boundary-driver.c").write_text(driver)
        metadata = [json.loads(line) for line in (root / "release.jsonl").read_text().splitlines()]
        groups = []
        for row in (row for row in metadata if "case" not in row):
            points = boundary_inputs(row)
            payload = (
                f"{row['func_id']} {row['dim']} {len(points)}\n"
                + "\n".join(" ".join(format(value, ".17g") for value in point) for point in points)
                + "\n"
            )
            groups.append((row["func_id"], payload.encode()))
        record["boundary_inputs"] = []
        outputs = []
        for kind, extra in (
            ("release", []),
            ("ubsan", ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"]),
        ):
            executable = root / f"boundary-{kind}"
            flags = [*record["compiler_flags"], *extra]
            compiled = subprocess.run(
                ["gcc", *flags, "boundary-driver.c", "-lm", "-o", str(executable)],
                cwd=root,
                capture_output=True,
                timeout=120,
                check=True,
            )
            (root / f"boundary-{kind}-compiler.log").write_bytes(compiled.stderr)
            combined = b""
            for fid, payload in groups:
                runs = [
                    subprocess.run(
                        [str(executable)],
                        input=payload,
                        cwd=root,
                        capture_output=True,
                        timeout=30,
                        check=True,
                    )
                    for _ in range(2)
                ]
                if runs[0].stdout != runs[1].stdout or runs[0].stderr or runs[1].stderr:
                    raise ValueError("Boundary source capture differed or emitted diagnostics")
                combined += runs[0].stdout
                if kind == "release":
                    (root / f"boundary-f{fid:02d}.txt").write_bytes(payload)
                    record["boundary_inputs"].append({"fid": fid, "sha256": digest(payload)})
            (root / f"boundary-{kind}.jsonl").write_bytes(combined)
            record[f"boundary_{kind}_sha256"] = digest(combined)
            record[f"boundary_{kind}_executable_sha256"] = digest(executable.read_bytes())
            record[f"boundary_{kind}_flags"] = flags
            outputs.append(combined)
        if outputs[0] != outputs[1]:
            raise ValueError("Release and UBSan boundary outputs differ")
        record["boundary_driver_sha256"] = digest(driver.encode())
        record["boundary_cases"] = len(outputs[0].splitlines())
    (output / "boundary-provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"boundary_cases": sum(row["boundary_cases"] for row in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
