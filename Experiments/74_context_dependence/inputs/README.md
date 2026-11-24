# External benchmark inputs

These two archives supply the external datasets in the [experiment protocol](../PROTOCOL.md).

- `quantumpioneer_kinetics.csv.gz` contains the exact SynRXN QuantumPioneer CSV projection pinned by the protocol. Decompression preserves the original CSV bytes, including the proposed splits and native targets. See the [source attribution and projection changes](QuantumPioneer2026.NOTICE.txt).
- `enzymemap_v2_brenda2023.csv.gz` is the original EnzymeMap v2 archive by Esther Heid, Daniel Probst, William H. Green and Georg K. H. Madsen. It is distributed as published, including the EC labels. Cite the [dataset](https://doi.org/10.5281/zenodo.8254726) and the [EnzymeMap paper](https://doi.org/10.1039/D3SC02048G) when using it.

Both datasets are distributed under [CC BY 4.0](CC-BY-4.0.txt).

From the root of the SynRFP checkout, verify the archives and unpack QuantumPioneer into the experiment's working directory. Keep EnzymeMap compressed to preserve its pinned archive checksum.

```sh
cd Experiments/74_context_dependence/inputs
sha256sum -c SHA256SUMS
mkdir -p ../../work/74_context_dependence/inputs
gzip -dc quantumpioneer_kinetics.csv.gz > ../../work/74_context_dependence/inputs/quantumpioneer_kinetics.csv
printf '%s  %s\n' \
  e81aa5f9158564e64356a524c046469e2a13e471469ff641713ded3c37ff1520 \
  ../../work/74_context_dependence/inputs/quantumpioneer_kinetics.csv | sha256sum -c -
```
