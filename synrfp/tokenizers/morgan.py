"""Graph-native count ECFP with physical-environment duplicate suppression.

The construction follows Rogers & Hahn, JCIM50 (2010),742–754,
https://doi.org/10.1021/ci100050t . The stable64-bit identifiers, graph attribute
profile and full-label propagation are SynRFP conventions, not RDKit bit IDs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import networkx as nx

from synrfp.graph.molecule import Molecule, NodeId
from synrfp.tokenizers.base import BaseTokenizer
from synrfp.tokenizers.utils import _h64

# Local atomic-number lookup; no RDKit import or runtime chemistry inference.
_ELEMENTS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni '
    'Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe '
    'Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au '
    'Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf '
    'Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
).split()
_ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(_ELEMENTS)}


def _freeze(value):
    """Stable, typed representation of the selected additional graph labels."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('ECFP attributes must be finite')
        return number
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    raise ValueError('Unsupported ECFP attribute type: ' + type(value).__name__)


def _integer(value, name, minimum=None):
    if value is None:
        value = 0
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + ' must be an integer')
    result = int(value)
    if not math.isfinite(float(value)) or result != value:
        raise ValueError(name + ' must be a finite integer')
    if minimum is not None and result < minimum:
        raise ValueError(name + ' is outside its supported domain')
    return result


def _element(attributes):
    symbol = attributes.get('element')
    number = _ATOMIC_NUMBERS.get(symbol) if isinstance(symbol, str) else None
    explicit = attributes.get('atomic_number')
    if explicit is not None:
        explicit = _integer(explicit, 'atomic_number', 0)
        if explicit > 118:
            raise ValueError('atomic_number exceeds118')
        if number is not None and explicit != number:
            raise ValueError('Conflicting element and atomic_number')
        if number is None and explicit:
            number = explicit
    # Synthetic/partial graphs remain usable, without pretending missing or
    # unknown elements are carbon or conflating two distinct unknown symbols.
    return ('Z', number) if number is not None else ('unknown_element', _freeze(symbol))


def _mass(attributes):
    isotope = _integer(attributes.get('isotope', 0), 'isotope', 0)
    if isotope:
        return ('isotope_mass_number', isotope)
    if attributes.get('mass') is not None:
        value = float(attributes['mass'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('mass must be positive and finite')
        return ('supplied_atomic_mass', value)
    return ('unspecified_isotope',)


@dataclass(frozen=True)
class MorganOccurrence:
    """An emitted occurrence, before reaction combination or sketching.

    Atom/bond IDs are witness handles and never enter the feature hash. At r=0
    every node has its own occurrence despite an empty bond support. Positive
    radii retain one representative per physical bond support. If equivalent
    candidates tie in token ID, the chosen center is only a witness convention.
    """
    token: int
    center: NodeId
    radius: int
    bonds: FrozenSet[Tuple[NodeId, NodeId]]


class MorganTokenizer(BaseTokenizer):
    """Graph-native ECFP occurrence counts, with structural duplicate removal.

    Fixed initial fields are heavy-neighbor degree, heavy bond valence, element
    (local atomic number), supplied isotope/mass, charge, total hydrogen count
    and graph-derived ring membership. ``node_attrs`` and ``edge_attrs`` add
    labels; they do not disable these fixed fields or the bond type.

    ``hcount`` denotes non-material attached H (SynKit's GetTotalNumHs default).
    Material H neighbors are added once to obtain total H. Material H nodes
    remain nodes if explicitly present: the input graph's H representation is
    part of the profile. No H suppression, valence completion, aromaticity/CIP
    assignment or other molecular normalization is performed here.

    Missing isotope/mass means unspecified isotope; the default SynKit graph
    loader does not retain isotope, so this tokenizer cannot recover it. Stereo
    is excluded by default. No claim of RDKit numeric-ID, universal support or
    count-vector parity is made: this uses its own64-bit hash and tie-breaking
    and updates all atom labels, rather than RDKit's dead-center optimization.
    """
    PROFILE = 'synrfp-ecfp-count-v1'
    SCHEMA = 'synrfp-morgan-ecfp-v004'
    FIXED_FIELDS = ('heavy_degree', 'heavy_valence', 'element', 'mass',
                    'charge', 'total_hydrogens', 'in_ring')

    def __init__(self, node_attrs: Optional[Sequence[str]] = None,
                 edge_attrs: Optional[Sequence[str]] = None) -> None:
        super().__init__(node_attrs=node_attrs, edge_attrs=edge_attrs)

    def __repr__(self):
        return (f'MorganTokenizer(node_attrs={self.node_attrs}, '
                f'edge_attrs={self.edge_attrs}, profile={self.PROFILE!r})')

    @staticmethod
    def describe():
        return ('from synrfp.tokenizers.morgan import MorganTokenizer\n'
                'tok = MorganTokenizer()\nC = tok.tokens_graph(graph, radius=2)\n')

    @staticmethod
    def _bond_order(attributes):
        order = attributes.get('order', 1)
        if isinstance(order, bool) or not isinstance(order, Real):
            raise ValueError('ECFP requires a numeric graph bond order')
        order = float(order)
        if not math.isfinite(order) or order < 0:
            raise ValueError('ECFP bond order must be finite and nonnegative')
        return 1.5 if attributes.get('aromatic') is True else order

    @classmethod
    def _bond_valence(cls, attributes):
        # Aromatic presentation order1.5 need not equal a heteroatom's Lewis
        # bond valence (e.g. two single bonds at pyrrolic[nH]). Use supplied
        # authoritative Lewis resources; never run a new perception algorithm.
        sigma, pi = attributes.get('sigma_order'), attributes.get('pi_order')
        if sigma is not None and pi is not None:
            values = (sigma, pi)
        elif attributes.get('kekule_order') is not None:
            values = (attributes['kekule_order'],)
        else:
            return cls._bond_order(attributes)
        if any(isinstance(x, bool) or not isinstance(x, Real)
               or not math.isfinite(float(x)) or float(x) < 0 for x in values):
            raise ValueError('ECFP Lewis bond resources must be finite and nonnegative')
        return sum(float(x) for x in values)

    @classmethod
    def _validate_graph(cls, G):
        for u, v in G.edges:
            if u not in G.nodes or v not in G.nodes:
                raise ValueError('ECFP bond endpoint missing from graph')
            if u == v:
                raise ValueError('ECFP requires a loop-free molecular graph')

    def atom_invariants(self, G: Molecule) -> Dict[NodeId, Tuple]:
        """Return seven fixed fields for profile/availability audits.

        Ring membership is derived from nonbridge edges, independent of atom IDs
        and of a selected cycle basis. Unknown elements retain an explicit tag.
        Heavy valence sums non-H incident sigma+pi resources, supplied Kekule
        order, or presentation bond order as fallback, in that order. This
        preserves available Lewis valence at aromatic heteroatoms.
        """
        if not isinstance(G, Molecule):
            raise TypeError('Expected Molecule')
        self._validate_graph(G)
        graph = nx.Graph()
        graph.add_nodes_from(G.nodes)
        graph.add_edges_from(G.edges)
        bridges = {frozenset(edge) for edge in nx.bridges(graph)}
        ring_atoms = {v for edge in G.edges if frozenset(edge) not in bridges for v in edge}
        elements = {v: _element(a) for v, a in G.nodes.items()}
        result = {}
        for v, attributes in G.nodes.items():
            neighbors = G.adj.get(v, ())
            hydrogens = sum(elements[u] == ('Z', 1) for u in neighbors)
            heavy = [u for u in neighbors if elements[u] != ('Z', 1)]
            valence = sum(self._bond_valence(G.edge_attr(v, u)) for u in heavy)
            hcount = _integer(attributes.get('hcount', 0), 'hcount', 0)
            charge = _integer(attributes.get('charge', 0), 'charge')
            result[v] = (len(heavy), valence, elements[v], _mass(attributes),
                         charge, hcount + hydrogens, v in ring_atoms)
        return result

    def tokenize_with_provenance(self, G: Molecule, radius: int):
        """Return ``(Counter, occurrences)`` before combining reaction sides.

        All atom labels update synchronously at every radius. A candidate's bond
        support expands by its incident bonds and previous neighbor supports.
        Previously emitted supports win over larger radii; within one radius,
        the minimum stable token ID represents a duplicate support. Equal tokens
        on different physical supports still contribute multiple occurrences.
        """
        super().tokens_graph(G, radius)
        fixed = self.atom_invariants(G)
        labels = {v: _h64(('ecfp0', self.PROFILE, fixed[v],
                          tuple((a, _freeze(G.nodes[v].get(a)))
                                for a in sorted(set(self.node_attrs)))))
                  for v in G.nodes}
        occurrences = [MorganOccurrence(labels[v], v, 0, frozenset()) for v in G.nodes]
        supports = {v: frozenset() for v in G.nodes}
        seen = set()
        edge_labels = {edge: (self._bond_order(attrs),
                              tuple((a, _freeze(attrs.get(a)))
                                    for a in sorted(set(self.edge_attrs))))
                       for edge, attrs in G.edges.items()}
        for r in range(1, radius + 1):
            updated = {}
            next_supports = {}
            candidates = []
            for v in G.nodes:
                neighborhood = []
                bonds = set(supports[v])
                for u in G.adj.get(v, ()):
                    edge = (v, u) if v < u else (u, v)
                    neighborhood.append((edge_labels[edge], labels[u]))
                    bonds.add(edge)
                    bonds.update(supports[u])
                # Typed selected attributes may differ; repr provides a total
                # deterministic order without relying on Python hash/node IDs.
                neighborhood.sort(key=lambda item: (item[0][0], repr(item[0][1]), item[1]))
                token = _h64(('ecfp', self.PROFILE, r, labels[v], tuple(neighborhood)))
                updated[v] = token
                next_supports[v] = frozenset(bonds)
                if bonds:
                    candidates.append(MorganOccurrence(token, v, r, frozenset(bonds)))
            for occurrence in sorted(candidates, key=lambda x: (x.token, x.center)):
                if occurrence.bonds not in seen:
                    occurrences.append(occurrence)
                    seen.add(occurrence.bonds)
            labels, supports = updated, next_supports
        return Counter(x.token for x in occurrences), occurrences

    def tokens_graph(self, G: Molecule, radius: int) -> Counter:
        return self.tokenize_with_provenance(G, radius)[0]
