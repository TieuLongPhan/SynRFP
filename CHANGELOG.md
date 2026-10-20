# Changelog

## 0.0.4

MorganTokenizer now implements graph-native count ECFP with fixed atom invariants
and physical bond-support duplicate suppression. Previous versions applied a
WL-like refinement under different hash tags, emitting every atom at every radius.
Morgan fingerprints and downstream sketches therefore change and require fresh,
versioned feature caches/results. Historical0.0.3 experiments remain historical.
WL/path tokenizers, reaction combination and sketch implementations are unchanged.

The initial profile includes heavy-neighbor degree, heavy bond valence, element,
available isotope/mass, charge, total attached H and graph-derived ring membership. Heavy valence uses supplied sigma+pi
resources or Kekule order before falling back to presentation bond order; this
retains available Lewis valence for aromatic heteroatoms.
Selected constructor attributes add information and cannot disable those fields.
Radius0 retains every atom occurrence; positive radii keep the earlier radius,
then the minimum stable64-bit identifier for the same physical bond support.
Identical features at distinct supports or in disconnected copies retain counts.
All labels update synchronously; only duplicate emissions are suppressed.

Profile: `synrfp-ecfp-count-v1`; schema: `synrfp-morgan-ecfp-v004`.
`tokenize_with_provenance` returns the Counter and an occurrence ledger containing
token, center, radius and physical bonds. `atom_invariants` exposes the seven fixed
fields for bounded validation. Center IDs are provenance, never hash input.

The algorithm follows Rogers and Hahn, *J. Chem. Inf. Model.*50(2010),742–754,
[doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t). It uses SynRFP's own
hash, graph profile and full-atom label propagation; it does not promise RDKit
identifier or universal count/support equivalence. Material H nodes remain if
present; no normalization or stereo inference is added. Missing isotope/mass
cannot be recovered, and the default SynKit graph loader omits isotope. The
unspecified-isotope profile is explicit; stereo remains off by default. Unknown
or synthetic element labels stay tagged unknown, never silently become carbon.
The tokenizer has no RDKit runtime dependency; RDKit is optional in audit tests.
