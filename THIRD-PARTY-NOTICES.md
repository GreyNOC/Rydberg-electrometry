# Third-party notices — RydSim v0.2.0 portable build

The portable `rydsim.exe` is built with PyInstaller and embeds the CPython
runtime together with the libraries below. Each is redistributed under its own
licence. PyInstaller is GPLv2 **with the bootloader exception**, which expressly
permits distributing the frozen application under any licence; no copyleft
obligation attaches to this binary.

RydSim itself is MIT — see `LICENSE`.

This list is generated from the SBOM of the exact locked build environment
(`packaging/requirements-build.lock`), not from the developer's global
interpreter, so it describes what actually ships.

| Package | Version | Licence |
|---|---|---|
| altgraph | 0.17.5 | MIT |
| arrow | 1.4.0 | License :: OSI Approved :: Apache Software License |
| attrs | 26.1.0 | MIT |
| boolean.py | 5.0 | BSD-2-Clause |
| chardet | 5.2.0 | License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+) |
| contourpy | 1.3.3 | License :: OSI Approved :: BSD License |
| cycler | 0.12.1 | License :: OSI Approved :: BSD License |
| cyclonedx-bom | 7.3.1 | Apache-2.0 |
| cyclonedx-python-lib | 11.11.1 | Apache-2.0 |
| defusedxml | 0.7.1 | Python-2.0 |
| fonttools | 4.63.0 | MIT |
| fqdn | 1.5.1 | License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0) |
| idna | 3.18 | BSD-3-Clause |
| isoduration | 20.11.0 | ISC |
| joblib | 1.5.3 | BSD-3-Clause |
| jsonpointer | 3.1.1 | License :: OSI Approved :: BSD License |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| kiwisolver | 1.5.0 | License :: OSI Approved :: BSD License |
| lark | 1.3.1 | MIT |
| license-expression | 30.4.4 | Apache-2.0 |
| lxml | 6.1.1 | BSD-3-Clause |
| matplotlib | 3.11.1 | Python-2.0 |
| narwhals | 2.24.0 | MIT |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packageurl-python | 0.17.6 | MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pefile | 2024.8.26 | MIT |
| pillow | 12.3.0 | MIT-CMU |
| pip | 26.2.1 | MIT |
| pip-requirements-parser | 32.0.1 | MIT |
| py-serializable | 2.1.0 | Apache-2.0 |
| pyinstaller | 6.22.0 | GPL-2.0-only |
| pyinstaller-hooks-contrib | 2026.6 | GPL-2.0-only |
| pyparsing | 3.3.2 | MIT |
| python-dateutil | 2.9.0.post0 | License :: OSI Approved :: Apache Software License |
| pywin32-ctypes | 0.2.3 | BSD-3-Clause |
| referencing | 0.37.0 | MIT |
| rfc3339-validator | 0.1.4 | MIT |
| rfc3986-validator | 0.1.1 | MIT |
| rfc3987-syntax | 1.1.0 | MIT |
| rpds-py | 2026.6.3 | MIT |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| scipy | 1.17.1 | License :: OSI Approved :: BSD License |
| setuptools | 65.5.0 | MIT |
| six | 1.17.0 | MIT |
| sortedcontainers | 2.4.0 | License :: OSI Approved :: Apache Software License |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| typing_extensions | 4.16.0 | PSF-2.0 |
| tzdata | 2026.3 | Apache-2.0 |
| uri-template | 1.3.0 | MIT |
| webcolors | 25.10.0 | BSD-3-Clause |

Numerical kernels: numpy and scipy bundle OpenBLAS (BSD-3-Clause). No
libgfortran/libquadmath is present in the bundle (verified at build time).

Machine-readable inventory: `rydsim-0.2.0-sbom.cdx.json` (CycloneDX 1.6).
