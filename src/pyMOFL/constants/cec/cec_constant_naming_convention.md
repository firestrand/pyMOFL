# CEC Benchmark Constant‑File Convention

---

## 1 . Directory layout (canonical)

```text
constants/
└── CEC
      ├──{YEAR}
        ├── meta_{YEAR}.json    # ✨ single manifest for the whole year (see § 4)
        └── f{FUNC:02d}/        # one folder per benchmark function
            ├── shift_D10.txt
            ├── rot_D10.txt
            ├── rot_nr_D10.txt
            ├── perm_D10.txt
            └── …
```

*Rationale — “year” and “function” already appear in the path; filenames stay concise. A single top‑level manifest avoids redundancy across \~30 function folders.*

Any file copied outside this hierarchy remains identifiable because each text file starts with a header comment—see § 3.

---

## 2 . File‑name template

```text
{DATATYPE}{VARIANT}_D{DIMENSION}.txt
```

| Token       | Allowed values                           | Notes                                        |
| ----------- | ---------------------------------------- | -------------------------------------------- |
| `DATATYPE`  | `shift`, `rot`, `perm`, `bias`           | Easily extensible—just add a new word.       |
| `VARIANT`   | (empty), `_ns`, `_nr`, `_sub` …          | Use sparingly; meanings in § 2.1 table.      |
| `DIMENSION` | integer, **no padding**, preceded by `D` | Length **of the vector/matrix in the file**. |

> **Single‑file model**  If a constant covers *any* runtime dimension ≤ `DIMENSION` (the usual “read‑what‑you‑need” pattern in CEC reference code), store **only one file at the maximum size**.
>
> *Example — a function defined for 10, 30 and 50 D may ship just `shift_D50.txt`; runners load N ≤ 50 components as required.*
>
> Dimension‑specific files are still allowed when they provide materially different data (e.g. separate rotation matrices for each D).

### 2.1 Variants glossary (keep updated)

| Variant | Meaning                                          |
| ------- | ------------------------------------------------ |
| `_ns`   | “no‑shift” vector (all zeros)                    |
| `_nr`   | “no‑rotation” / identity matrix                  |
| `_sub`  | sub‑component matrix used by composite functions |

> **Examples inside `…/input_data/f05/`**
>
> * `shift_D50.txt` (single‑file model, reused for all smaller dims)
> * `rot_D50.txt`
> * `rot_nr_D50.txt`
> * `perm_D50.txt`

---

## 3 . File header (first line)

Every text file **must begin** with a single‑line comment so it remains self‑describing when detached from the tree:

```txt
# CEC2021  •  f05  •  shift vector  •  dim 50
0.12345  -1.2345  …
```

*Use `#` for comments—C and Python readers ignore it.*

---

## 4 . Year‑wide manifest (`meta_{YEAR}.json`)

Stored at `input_data/meta_{YEAR}.json`, this single JSON file enumerates every benchmark function, its supported dimensions, and the relative paths of its constant files.

```json
{
  "year": 2021,
  "functions": {
    "f05": {
      "dimensions": [10, 30, 50],
      "files": {
        "shift": "f05/shift_D50.txt",
        "rot":   "f05/rot_D50.txt",
        "perm":  "f05/perm_D50.txt"
      }
    },
    "f06": {
      "dimensions": [10, 30, 50],
      "files": {
        "shift": "f06/shift_D50.txt",
        "rot":   "f06/rot_D50.txt",
        "perm":  "f06/perm_D50.txt"
      }
    }
    /* …repeat for all functions… */
  }
}
```

*Why a single manifest?*  Tools can parse one file, build lookup tables, and validate completeness in a single pass—no directory walking required.

---

### Summary

* **No redundant tokens** – year & function live in the path.
* **Human‑obvious** – `shift_D50.txt` needs no cheat‑sheet.
* **Script‑friendly** – fixed pattern + global manifest = painless I/O.
* **Forward‑proof** – adding CEC 2025 or a new datatype is trivial.

Refactor once and you’ll never worry about file chaos again.
