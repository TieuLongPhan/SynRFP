"""Exact rooted certificates on SynKit's configured colored incidence graph."""
from __future__ import annotations

from hashlib import sha256
import json

import igraph
from synkit.Graph.Canon.exact import _encode_color
from synkit.Graph.Stereo.canonical import AtomVertex, BondVertex
from synkit.Graph.Stereo.configured import expand_configured_stereograph


class ConfiguredExpansion:
    def __init__(self, graph, descriptors, atom_color, bond_color):
        expanded=expand_configured_stereograph(graph,descriptors,atom_color=atom_color,bond_color=bond_color)
        self.ids={node:i for i,node in enumerate(expanded)}
        colors=[_encode_color(expanded.nodes[n]["color"]) for n in self.ids]
        self.palette=sorted(set(colors))
        ranks={color:i for i,color in enumerate(self.palette)}
        self.base_colors=[2*ranks[color] for color in colors]
        self.edges=[(self.ids[l],self.ids[r]) for l,r in expanded.edges]
        self.graph=igraph.Graph(n=len(self.ids),edges=self.edges,directed=False)
        self.bonds={
            frozenset(neighbor.reference for neighbor in expanded[node] if isinstance(neighbor,AtomVertex)):node
            for node in self.ids if isinstance(node,BondVertex)
        }

    def digest(self, vertex):
        # Mark the selected incidence vertex without changing its chemical
        # color. This is equivalent to adding the root color before expansion.
        colors=self.base_colors.copy()
        colors[self.ids[vertex]]+=1
        permutation=self.graph.canonical_permutation(color=colors)
        canonical_colors=[0]*len(colors)
        for old,new in enumerate(permutation):
            canonical_colors[new]=colors[old]
        edges=sorted(tuple(sorted((permutation[a],permutation[b]))) for a,b in self.edges)
        code=json.dumps(("synkit-rooted-configuration-bliss-v1",self.palette,canonical_colors,edges),
                        separators=(",",":"))
        return sha256(code.encode("utf-8")).hexdigest()

    def atom(self, owner):
        return self.digest(AtomVertex(owner))

    def bond(self, left, right):
        return self.digest(self.bonds[frozenset((left,right))])
