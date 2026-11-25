# WL radius-2 token profiles

All profiles use the study 65 WL attributes: element, aromaticity, charge and
hydrogen count on atoms, and order on bonds. Reaction agents are included on
both sides. The sum of the component token multisets supplies the side counts.
Exact D is product minus reactant, and exact U is their sum. The release checks
all ordinary D, U, and paired keys against the original study 74 archive.

- **Ordinary** uses the published attribute set.
- **Stereo tetra** appends the rooted configured stereograph digest to the
  atom label at each registered tetrahedral centre. The root's component uses
  its complete tetrahedral registry. This extends the alcohol probe's rooted
  identity to multicomponent reactions.
- **Stereo E/Z** retains the tetrahedral labels and adds the corresponding
  exact rooted label to each registered planar double bond. Each bond digest
  uses its own planar descriptor and the connected constitution, with that
  bond marked. Missing descriptors supply an unset attribute in both profiles.
  The audit lists the source registry families.

SynKit expands the configured molecular graph and its descriptors into the
same typed colored incidence graph used by the native exact canonicalizer.
The accelerated runner marks the atom or bond incidence vertex, computes an
exact Bliss canonical permutation and serializes the full colored adjacency.
The root color preserves the original chemical color. The certificate format
has its own identifier and digest bytes. It induces the same rooted configured
equivalence as marking the atom or bond before SynKit's native canonicalization.
Native-versus-compiled controls check equality classes, renumbering and the
two alcohol pairs from Request 74.

A rooted molecule digest can change when chemistry elsewhere in that molecule
changes. Thus the label can encode surrounding structure along with stereo.
An oracle change on the full cohort measures the complete enriched
representation. The 15 constitution-matched reactions directly check the
effect on the selected stereo-only cases.
