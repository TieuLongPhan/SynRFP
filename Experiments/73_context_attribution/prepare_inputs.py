"""Rebuild the exact E1 input tables from pinned author sources."""

import argparse
import csv
import gzip
import hashlib
import io
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from openpyxl import load_workbook

SOURCE_REVISION = "d9e6b87ce1b881978490d68bfc00021e3b48127a"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/{SOURCE_REVISION}/"
)
SOURCES = {
    "bh.xlsx": (
        "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx",
        "d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e",
    ),
    "suzuki_source.xlsx": (
        "data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx",
        "a869e020ba31bd5676c67a4791c3b7384711b5216de6af444b8cd0a24c284640",
    ),
    "suzuki_0.tsv": (
        "data/Suzuki-Miyaura/random_splits/random_split_0.tsv",
        "d2acd54d3958a5369346177834e6c2b16388813bb28b105181a252a15c4a79a8",
    ),
}
OUTPUT_SHA256 = {
    "bh_classic.csv.gz": "ab5cea6673bb6c1e0dd13f200b25fabcecf8f13ad5471d09f3004f3712fea2f7",
    "suzuki_native_reference.csv.gz": "c5a3354079455ad37462253861a76d04e724911b9fc23929f3b10f51ad3a29ce",
}
SUZUKI_RESPONSE = "Product_Yield_PCT_Area_UV"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def materialize_source(name, source_dir):
    remote, expected = SOURCES[name]
    if source_dir is None:
        with urlopen(SOURCE_URL + remote, timeout=60) as response:
            data = response.read()
    else:
        data = (source_dir / name).read_bytes()
    if digest(data) != expected:
        raise ValueError(f"Unexpected author source checksum for {name}")
    return data


def sheet(data, name):
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        rows = workbook[name].iter_rows(values_only=True)
        header = next(rows)
        return [
            dict(zip(header, values))
            for values in rows
            if any(v is not None for v in values)
        ]
    finally:
        workbook.close()


def buchwald_hartwig_rows(source):
    reference = "FullCV_01"
    rows = sheet(source, reference)
    if len(rows) != 3955:
        raise ValueError("Unexpected classic Buchwald–Hartwig row count")
    return [
        {
            "sample_id": f"bh_classic:{position:04d}",
            "reference_sheet": reference,
            "reference_position": position,
            "ligand_smiles": row["Ligand"],
            "additive_smiles": row["Additive"],
            "base_smiles": row["Base"],
            "aryl_halide_smiles": row["Aryl halide"],
            "yield_pct": row["Output"],
            "source_doi": "10.1126/science.aar5169",
            "release_status": "local_reference_only_upstream_terms_not_requalified",
        }
        for position, row in enumerate(rows)
    ]


def suzuki_rows(workbook, tsv):
    measurements = sheet(workbook, "Sheet1")
    reactions = list(csv.DictReader(io.StringIO(tsv.decode("utf-8")), delimiter="\t"))
    indices = [int(row[""]) for row in reactions]
    if len(measurements) != 5760 or sorted(indices) != list(range(5760)):
        raise ValueError("Unexpected Suzuki source population")
    result = []
    for index, reaction in zip(indices, reactions, strict=True):
        row = measurements[index]
        measured = float(row[SUZUKI_RESPONSE])
        if not math.isclose(
            float(reaction["y"]) * 100, measured, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(f"Suzuki yield mismatch at author source index {index}")
        result.append(
            {
                "sample_id": "suzuki_miyaura:" + str(row["Reaction_No"]).strip(),
                "source_index": index,
                "author_reaction_smiles": reaction["rxn"],
                "author_response_fraction": reaction["y"],
                "local_yield_pct": format(measured, ".12g"),
                "measurement_method": SUZUKI_RESPONSE,
                "release_status": "local_reference_only_existing_permission_status_preserved",
            }
        )
    return result


def write_csv(path, rows):
    with (
        path.open("wb") as file,
        gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Read the three pinned, cached author files from this directory",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    sources = {name: materialize_source(name, args.source_dir) for name in SOURCES}
    tables = {
        "bh_classic.csv.gz": buchwald_hartwig_rows(sources["bh.xlsx"]),
        "suzuki_native_reference.csv.gz": suzuki_rows(
            sources["suzuki_source.xlsx"], sources["suzuki_0.tsv"]
        ),
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".e1-inputs-", dir=args.output_dir.parent
    ) as staging:
        directory = Path(staging)
        directory.chmod(0o700)
        for name, rows in tables.items():
            path = directory / name
            write_csv(path, rows)
            if digest(path.read_bytes()) != OUTPUT_SHA256[name]:
                raise ValueError(f"Rebuilt {name} does not match the pinned E1 input")
            print(f"{name}: {len(rows)} rows, SHA-256 {OUTPUT_SHA256[name]}")
        (directory / "SHA256SUMS").write_text(
            "".join(f"{checksum}  {name}\n" for name, checksum in OUTPUT_SHA256.items())
        )
        directory.rename(args.output_dir)


if __name__ == "__main__":
    main()
