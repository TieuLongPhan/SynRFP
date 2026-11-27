# Fingerprint agreement after SMILES rewriting and reversal

This audit used 2,000 reactions per endpoint sampled without replacement with seed 0. All seven arms processed the same ordered originals, randomized component and SMILES forms, and reactant–product reversals. The baseline is the benchmark release's n1 inputs. Archived original validity masks were replayed exactly. Classical vectors replayed exactly and the CPU transformer vectors were checked numerically against the released archives.

| Fingerprint | Rewriting exact agreement across endpoints | Reversal exact agreement across endpoints |
|---|---:|---:|
| `synrfp-wl-r2-b2048-parity-delta` | 100.00%–100.00% | 0.00%–0.20% |
| `synrfp-wl-r2-b2048-parity-delta_union` | 100.00%–100.00% | 0.00%–0.20% |
| `drfp` | 8.05%–42.85% | 100.00%–100.00% |
| `rdkit_diff` | 100.00%–100.00% | 0.00%–0.25% |
| `rdkit_struct` | 100.00%–100.00% | 0.05%–0.35% |
| `crxnfp` | 100.00%–100.00% | 0.00%–0.15% |
| `rxnfp_mlm` | 0.00%–0.00% | 0.00%–0.15% |
| `drfp_on_canonicalised_input` | 100.00%–100.00% | 100.00%–100.00% |

For the released DRFP encoder, exact rewriting agreement was 8.05%–42.85%. With canonicalization applied to each input at encoding time it was 100.00%–100.00%. The released encoder receives n1 strings directly. The canonicalized-input row measures an extra preprocessing step.

The SynRFP default encoded 7,999/8,000 original inputs. RXNFP-MLM and cRXNFP encoded 8,000/8,000 and 8,000/8,000, respectively. The failure ledger contains 9 fingerprint-by-form refusals across the seven arms and the optional DRFP control. The warning table contains 42 successful encodings with diagnostics. The RDKit reaction fingerprint code path accepts unsanitized components. SynRFP uses sanitized molecular graphs. A DRFP vector can be returned despite a component parse diagnostic. The transformer checkpoints have native 512-token truncation, which is counted as a warning when encoding succeeds.

The shared-valid columns exclude failed pairs from agreement, similarity and the absolute 1e−6 comparison. The release also retains the historical unconditional score columns. Equal all-zero vectors have similarity one. Transformer numbers are for one consistent CPU inference profile with each reaction and its two variants in the same batch.

## Encoders

- `synrfp-wl-r2-b2048-parity-delta` uses WL radius 2, parity seed 1, signed difference, 2048 bits.
- `synrfp-wl-r2-b2048-parity-delta_union` uses WL radius 2, parity seed 1, signed difference 1024 bits plus pooled counts 1024 bits.
- `drfp` uses released n1 source strings and native DRFP radius 3, rings, 2048 bits.
- `drfp_on_canonicalised_input` uses DRFP radius 3, rings, 2048 bits after RDKit canonical SMILES and sorting components within each field.
- `rdkit_diff` uses ReactionFromSmarts(useSmiles=True), native 2048-dimensional signed counts.
- `rdkit_struct` uses ReactionFromSmarts(useSmiles=True), native 4096-bit structural fingerprint.
- `crxnfp` uses checkpoint-45600 with official canonical preprocessing, masked mean pooling, native 512-token truncation.
- `rxnfp_mlm` uses bert_pretrained, agent-to-reactant native text adapter, CLS embedding, native 512-token truncation.

See [paired results](perturbation_all_fps.csv), [failure counts](encoding_failures.csv) and [encoding warnings](encoding_warnings.csv).
