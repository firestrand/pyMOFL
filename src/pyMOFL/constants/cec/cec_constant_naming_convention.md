Your draft is already excellent—clear, concise, and script-friendly. Here’s how to update it to reflect your new decisions:

---

## **Key Updates**

1. **Hybrid Naming for Matrices and Vectors**  
   - Use `matrix_{purpose}_D{dim}.txt` and `vector_{purpose}_D{dim}.txt` for all matrix/vector data, where `{purpose}` is a short, meaningful descriptor (e.g., `rotation`, `permutation`, `shift`, `data`, `hybrid`, etc.).
   - This makes the file’s role explicit and extensible.

2. **File Name Template**  
   - Update the template and table to reflect `matrix_` and `vector_` prefixes, and clarify `{purpose}`.

3. **Examples**  
   - Add examples for both matrix and vector files.

---

## **Updated Draft**

---

# CEC Benchmark Constant‑File Convention

---

## 1 . Directory layout (canonical)

```text
constants/
└── CEC
      ├──{YEAR}
        ├── meta_{YEAR}.json    # ✨ single manifest for the whole year (see § 4)
        └── f{FUNC:02d}/        # one folder per benchmark function
            ├── vector_shift_D10.txt
            ├── matrix_rotation_D10.txt
            ├── matrix_rotation_nr_D10.txt
            ├── matrix_permutation_D10.txt
            └── …
```

*Rationale — “year” and “function” already appear in the path; filenames stay concise. A single top‑level manifest avoids redundancy across \~30 function folders.*

Any file copied outside this hierarchy remains identifiable because each text file starts with a header comment—see § 3.

---

## 2 . File‑name template

```text
{TYPE}_{PURPOSE}{VARIANT}_D{DIMENSION}.txt
```

| Token       | Allowed values                           | Notes                                        |
| ----------- | ---------------------------------------- | -------------------------------------------- |
| `TYPE`      | `vector`, `matrix`                       | Required; clarifies the data structure.      |
| `PURPOSE`   | `shift`, `rotation`, `permutation`, `data`, `hybrid`, ... | Short, meaningful descriptor of the file’s role. |
| `VARIANT`   | (empty), `_ns`, `_nr`, `_sub`, ...       | Use sparingly; meanings in § 2.1 table.      |
| `DIMENSION` | integer, **no padding**, preceded by `D` | Length **of the vector/matrix in the file**. |

> **Single‑file model**  If a constant covers *any* runtime dimension ≤ `DIMENSION` (the usual “read‑what‑you‑need” pattern in CEC reference code), store **only one file at the maximum size**.
>
> *Example — a function defined for 10, 30 and 50 D may ship just `vector_shift_D50.txt`; runners load N ≤ 50 components as required.*
>
> Dimension‑specific files are still allowed when they provide materially different data (e.g. separate rotation matrices for each D).

### 2.1 Variants glossary (keep updated)

| Variant | Meaning                                          |
| ------- | ------------------------------------------------ |
| `_ns`   | “no‑shift” vector (all zeros)                    |
| `_nr`   | “no‑rotation” / identity matrix                  |
| `_sub`  | sub‑component matrix used by composite functions |

> **Examples inside `…/f05/`**
>
> * `vector_shift_D50.txt` (single‑file model, reused for all smaller dims)
> * `matrix_rotation_D50.txt`
> * `matrix_rotation_nr_D50.txt`
> * `matrix_permutation_D50.txt`
> * `matrix_data_D50.txt` (for generic data matrices)
> * `matrix_shift_and_rotation_D50.txt` (for files with both shift and rotation data, if needed)

---

## 3 . Year‑wide manifest (`meta_{YEAR}.json`)

Stored at `input_data/meta_{YEAR}.json`, this single JSON file enumerates every benchmark function, its supported dimensions, and the relative paths of its constant files.

```json
{
  "year": 2021,
  "functions": {
    "f05": {
      "dimensions": [10, 30, 50],
      "files": {
        "vector_shift": "f05/vector_shift_D50.txt",
        "matrix_rotation":   "f05/matrix_rotation_D50.txt",
        "matrix_permutation":  "f05/matrix_permutation_D50.txt"
      }
    },
    "f06": {
      "dimensions": [10, 30, 50],
      "files": {
        "vector_shift": "f06/vector_shift_D50.txt",
        "matrix_rotation":   "f06/matrix_rotation_D50.txt",
        "matrix_permutation":  "f06/matrix_permutation_D50.txt"
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
* **Human‑obvious** – `vector_shift_D50.txt` needs no cheat‑sheet.
* **Script‑friendly** – fixed pattern + global manifest = painless I/O.
* **Forward‑proof** – adding CEC 2025 or a new datatype is trivial.

Refactor once and you’ll never worry about file chaos again.

---

**Let me know if you want this as a markdown file, or if you want to add more examples or edge cases!**
