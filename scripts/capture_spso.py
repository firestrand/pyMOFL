"""Explicit source-backed native-coordinate SPSO capture; no optimizer execution."""

import argparse
import difflib
import hashlib
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


DRIVER = r"""
static void array_json(const double *values, int n) {
  printf("[");
  for (int d=0;d<n;d++) printf("%s%.17g",d?",":"",values[d]);
  printf("]");
}
int main(void) {
  E=exp((long double)1); pi=acos((long double)-1);
  const int ids[]={4,11,18,21};
  for(int i=0;i<4;i++) {
    int fid=ids[i]; struct problem pb=problemDef(fid);
    if(pb.SS.D!=(fid==4?2:fid==11?42:fid==18?4:3)) return 2;
    NATIVE_ADAPTER
    printf("{\"func_id\":%d,\"dim\":%d,\"objective\":%.17g,\"low\":",fid,pb.SS.D,pb.objective);
    array_json(pb.SS.min,pb.SS.D); printf(",\"high\":"); array_json(pb.SS.max,pb.SS.D);
    printf(",\"steps\":"); array_json(pb.SS.q.q,pb.SS.D); printf("}\n");
    for(int k=0;k<(fid==4||fid==18?3:2);k++) {
      struct position x={0}; x.size=pb.SS.D;
      for(int d=0;d<x.size;d++) x.x[d]=k==0?pb.SS.min[d]:pb.SS.max[d];
      if(k==2 && fid==4) {x.x[0]=0;x.x[1]=-50;}
      if(k==2 && fid==18) {x.x[0]=16;x.x[1]=19;x.x[2]=43;x.x[3]=49;}
      struct position q=quantis(x,pb.SS);
      double value=perf(q,fid,pb.SS,pb.objective);
      if(!isfinite(value)) return 3;
      printf("{\"func_id\":%d,\"dim\":%d,\"case\":\"%s\",\"x\":",fid,x.size,k==0?"lower":k==1?"upper":"documented_solution");
      array_json(x.x,x.size);printf(",\"quantized_x\":");array_json(q.x,q.size);
      printf(",\"value\":%.17g}\n",value);
    }
  }
  return 0;
}
"""


def prepare(archive_root: Path, output: Path) -> None:
    result = []
    for year in (2007, 2011):
        stem = "standard_pso_2007" if year == 2007 else "standard_pso_2011_c"
        archive = archive_root / (stem + ".zip")
        source = output / "originals" / str(year)
        expected = (
            "f9524f7f9568009b4ab5c76cd32d91c255fef978b4ff64891b460cb520f34bd1"
            if year == 2007
            else "11692f658158b18aafd97d667eeebdc7527cf21147d530d73d4c7eb795af0557"
        )
        if sha(archive.read_bytes()) != expected:
            raise ValueError("Unapproved source archive")
        with zipfile.ZipFile(archive) as zipped:
            members: dict[str, bytes] = {}
            for name in zipped.namelist():
                if not name.endswith((".c", ".h")):
                    continue
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or path.name in members:
                    raise ValueError("Unsafe or duplicate archive source member")
                members[path.name] = zipped.read(name)
        source.mkdir(parents=True, exist_ok=True)
        for filename, data in members.items():
            (source / filename).write_bytes(data)
        target = output / str(year)
        target.mkdir(parents=True, exist_ok=True)
        files = []
        for path in sorted(source.iterdir()):
            shutil.copyfile(path, target / path.name)
            files.append({"path": path.name, "sha256": sha(path.read_bytes())})
        patches = []
        if year == 2011:
            for filename in ("main.h", "problemDef.c"):
                path = target / filename
                old = path.read_text()
                if filename == "main.h":
                    new = "\n".join(line for line in old.splitlines() if "gsl" not in line) + "\n"
                else:
                    if old.count("if(pb.SS.normalise>0)") != 1:
                        raise ValueError("Unexpected native-coordinate patch site")
                    new = old.replace(
                        "if(pb.SS.normalise>0)",
                        "if(0) /* Native-coordinate harness: preserve declared quanta. */",
                    )
                path.write_text(new)
                patches.extend(
                    difflib.unified_diff(
                        old.splitlines(True),
                        new.splitlines(True),
                        fromfile="original/" + filename,
                        tofile="native/" + filename,
                    )
                )
            # Extract the final quantis definition exactly; do not copy its prototype.
            pso = (source / "PSO.c").read_text()
            start = pso.index("struct position quantis (struct position x, struct SS SS )")
            (target / "quantis.c").write_text(pso[start:])
            prefix = """#include "main.h"
struct result PSO(struct param param, struct problem problem) { abort(); }
double alea_normal(double mean, double std_dev, int option) { abort(); }
#include "perf.c"
#include "problemDef.c"
#include "tools.c"
#include "quantis.c"
"""
            native = "pb.SS.normalise=0;"
        else:
            prefix = '#define main spso_optimizer_main\n#include "main.c"\n#undef main\n'
            native = "/* 2007 constructor already uses native coordinates. */"
        driver = prefix + DRIVER.replace("NATIVE_ADAPTER", native)
        (target / "driver.c").write_text(driver)
        (target / "native.patch").write_text("".join(patches))
        result.append(
            {
                "year": year,
                "archive_sha256": expected,
                "source_url": "https://www.particleswarm.info/" + stem + ".zip",
                "source_files": files,
                "driver_sha256": sha(driver.encode()),
                "patch_sha256": sha("".join(patches).encode()),
                "rights": "unspecified; external local captures only",
                "coordinate_contract": "native; official quantis then perf distance-to-objective",
                "known_discrepancy": "2007 spring g2-positive branch uses stress g1 in multiplier; unmodified"
                if year == 2007
                else "2011 source corrects spring g2 multiplier",
            }
        )
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")


def capture(output: Path) -> None:
    records: list[dict[str, Any]] = json.loads((output / "preparation.json").read_text())
    for row in records:
        root = output / str(row["year"])
        flags = [
            "-std=c99",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Wformat=2",
        ]
        outputs = []
        for kind, extra in [
            ("release", []),
            ("ubsan", ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"]),
        ]:
            executable = root / kind
            compiled = subprocess.run(
                ["gcc", *flags, *extra, "driver.c", "-lm", "-o", str(executable)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            (root / (kind + "-compiler.log")).write_text(compiled.stderr)
            compiled.check_returncode()
            runs = [
                subprocess.run(
                    [str(executable)], cwd=root, capture_output=True, timeout=30, check=True
                )
                for _ in range(2)
            ]
            if runs[0].stdout != runs[1].stdout:
                raise ValueError("Source capture was not repeatable")
            if runs[0].stderr or runs[1].stderr:
                raise ValueError("Source capture emitted runtime diagnostics")
            (root / (kind + ".jsonl")).write_bytes(runs[0].stdout)
            outputs.append(runs[0].stdout)
            row[kind + "_sha256"] = sha(runs[0].stdout)
            row[kind + "_executable_sha256"] = sha(executable.read_bytes())
            row[kind + "_compiler_flags"] = [*flags, *extra]
        if outputs[0] != outputs[1]:
            raise ValueError("Release and UBSan source outputs differ")
        decoded = [json.loads(line) for line in outputs[0].splitlines()]
        if len(decoded) != 14:
            raise ValueError("Source capture has incomplete selected coverage")
        if not all(
            r["value"] == 0
            for r in decoded
            if r.get("case") == "documented_solution" and r["func_id"] == 4
        ):
            raise ValueError("Source Tripod optimum control failed")
        row["compiler_flags"] = flags
        row["compiler"] = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout.splitlines()[0]
        row["platform"] = platform.platform()
    (output / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(CHECKOUT):
        parser.error("Reference output must be outside the repository")
    if output.exists() and any(output.iterdir()):
        parser.error("Reference output directory must be empty")
    prepare(args.archives.resolve(), output)
    capture(output)
    print(json.dumps({"versions": 2, "captured_cases": 20, "metadata_records": 8}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
