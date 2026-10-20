"""Count/support contracts independent of a particular fingerprint hash."""
from collections import Counter, deque
import builtins
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from synrfp.graph.molecule import Molecule
from synrfp.tokenizers.morgan import MorganTokenizer, _ATOMIC_NUMBERS


def molecule(n, edges, elements=None):
    nodes={i:{'element':(elements or {}).get(i,'C'),'charge':0,'aromatic':False} for i in range(n)}
    graph=Molecule.from_dicts(nodes,{e:{'order':1} for e in edges})
    for v,a in graph.nodes.items():a['hcount']=max(0,4-graph.degree(v)) if a['element']=='C' else 0
    return graph


def bfs_supports(graph, radius):
    """Independent radius-ball edge oracle; no recursive tokenizer state/hash."""
    occurrences=[];seen=set()
    for v in graph.nodes:occurrences.append((0,frozenset(),v))
    for r in range(1,radius+1):
        candidates=set()
        for v in graph.nodes:
            distance={v:0};queue=deque([v])
            while queue:
                w=queue.popleft()
                if distance[w]>=r:continue
                for u in graph.adj.get(w,()):
                    if u not in distance:distance[u]=distance[w]+1;queue.append(u)
            edges=frozenset(e for e in graph.edges if min(distance.get(e[0],math.inf),distance.get(e[1],math.inf))<r)
            if edges and edges not in seen:candidates.add(edges)
        occurrences.extend((r,e,None) for e in candidates);seen.update(candidates)
    return Counter((r,e) for r,e,_ in occurrences)


class ECFPControls(unittest.TestCase):
    def test_radius0_does_not_merge_empty_bondsets_or_isolated_atoms(self):
        graph=molecule(4,[]);tokens,ledger=MorganTokenizer().tokenize_with_provenance(graph,5)
        self.assertEqual(sum(tokens.values()),4);self.assertEqual(len(ledger),4);self.assertTrue(all(x.radius==0 for x in ledger))
    def test_same_physical_bond_deduplicates_but_distinct_copies_count(self):
        tokenizer=MorganTokenizer();single=molecule(2,[(0,1)]);copies=molecule(4,[(0,1),(2,3)])
        for radius in range(1,5):
            a=tokenizer.tokens_graph(single,radius);b=tokenizer.tokens_graph(copies,radius)
            self.assertEqual(sum(a.values()),3);self.assertEqual(b,Counter({k:2*v for k,v in a.items()}))
    def test_triangle_saturates_and_retains_distinct_supports(self):
        graph=molecule(3,[(0,1),(1,2),(0,2)]);tokens,ledger=MorganTokenizer().tokenize_with_provenance(graph,6)
        self.assertEqual(Counter(x.radius for x in ledger),Counter({0:3,1:3,2:1}));self.assertEqual(sum(tokens.values()),7)
        positive=[x.bonds for x in ledger if x.radius];self.assertEqual(len(positive),len(set(positive)))
    def test_independent_bfs_support_oracle_on_finite_graph_catalogue(self):
        edges=[[],[(0,1)],[(0,1),(1,2),(2,3)],[(0,1),(0,2),(0,3),(3,4)],[(0,1),(1,2),(0,2),(2,3),(3,4)],[(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,3)]]
        for edge_list in edges:
            graph=molecule(max((max(e) for e in edge_list),default=0)+1,edge_list)
            for r in range(5):
                tokens,ledger=MorganTokenizer().tokenize_with_provenance(graph,r)
                actual=Counter((x.radius,x.bonds) for x in ledger)
                self.assertEqual(actual,bfs_supports(graph,r));self.assertEqual(sum(tokens.values()),sum(actual.values()))
    def test_node_edge_insertion_renumbering_and_map_provenance_invariance(self):
        graph=molecule(7,[(0,1),(1,2),(2,0),(1,3),(3,4),(4,5),(4,6)],{3:'N',6:'O'});tokenizer=MorganTokenizer();expected,ledger=tokenizer.tokenize_with_provenance(graph,4)
        for seed in range(10):
            rng=random.Random(seed);permutation=rng.sample(list(range(100,107)),7);mapping=dict(zip(graph.nodes,permutation));nodes=list(graph.nodes);edges=list(graph.edges);rng.shuffle(nodes);rng.shuffle(edges)
            g=Molecule.from_dicts({mapping[v]:{**graph.nodes[v],'atom_map':seed*100+v} for v in nodes},{(mapping[u],mapping[v]):dict(graph.edges[u,v]) for u,v in edges})
            actual,witnesses=tokenizer.tokenize_with_provenance(g,4);self.assertEqual(actual,expected)
            transported=Counter((x.radius,frozenset(tuple(sorted((mapping[u],mapping[v]))) for u,v in x.bonds),x.token) for x in ledger)
            self.assertEqual(Counter((x.radius,x.bonds,x.token) for x in witnesses),transported)
    def test_fixed_ring_and_chemical_fields_cannot_be_disabled_by_attrs(self):
        tokenizer=MorganTokenizer(node_attrs=[],edge_attrs=[])
        cycle=molecule(3,[(0,1),(1,2),(0,2)]);chain=molecule(3,[(0,1),(1,2)])
        self.assertTrue(tokenizer.atom_invariants(cycle)[1][-1]);self.assertFalse(tokenizer.atom_invariants(chain)[1][-1])
        atom=Molecule.from_dicts({0:dict(element='C',hcount=4)},{});changed=Molecule.from_dicts({v:dict(a) for v,a in atom.nodes.items()},atom.edges);changed.nodes[0]['charge']=1
        self.assertNotEqual(tokenizer.tokens_graph(atom,0),tokenizer.tokens_graph(changed,0))
        changed=Molecule.from_dicts({v:dict(a) for v,a in atom.nodes.items()},atom.edges);changed.nodes[0]['isotope']=13
        self.assertNotEqual(tokenizer.tokens_graph(atom,0),tokenizer.tokens_graph(changed,0))
        self.assertEqual(tokenizer.PROFILE,'synrfp-ecfp-count-v1');self.assertEqual(len(tokenizer.FIXED_FIELDS),7)
    def test_explicit_hydrogen_count_and_heavy_valence(self):
        graph=Molecule.from_dicts({0:dict(element='C',hcount=3),1:dict(element='H',hcount=0)}, {(0,1):{'order':1}})
        fixed=MorganTokenizer().atom_invariants(graph)[0]
        self.assertEqual((fixed[0],fixed[1],fixed[5]),(0,0,4))
        self.assertEqual(sum(MorganTokenizer().tokens_graph(graph,0).values()),2) # MaterialH remains an explicit graph node.
    def test_supplied_lewis_valence_handles_aromatic_pyrrolic_nitrogen(self):
        graph=Molecule.from_dicts({0:dict(element='N',aromatic=True,hcount=1),1:dict(element='C'),2:dict(element='C')},
            {(0,1):dict(order=1.5,aromatic=True,sigma_order=1,pi_order=0,kekule_order=1),
             (0,2):dict(order=1.5,aromatic=True,sigma_order=1,pi_order=0,kekule_order=1)})
        self.assertEqual(MorganTokenizer().atom_invariants(graph)[0][1],2)
        graph.edges[0,1].pop('sigma_order');graph.edges[0,1].pop('pi_order')
        self.assertEqual(MorganTokenizer().atom_invariants(graph)[0][1],2)
    def test_incomplete_sigma_does_not_override_double_bond_order(self):
        graph=Molecule.from_dicts({0:dict(element='C'),1:dict(element='O')},{(0,1):dict(order=2,sigma_order=1)})
        self.assertEqual(MorganTokenizer().atom_invariants(graph)[0][1],2)
    def test_equivalent_kekule_ring_assignments_keep_default_tokens(self):
        graph=molecule(6,[(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)])
        for v in graph.nodes:graph.nodes[v].update(aromatic=True,hcount=1)
        for index,edge in enumerate(graph.edges):graph.edges[edge].update(order=1.5,aromatic=True,kekule_order=1+index%2,sigma_order=1,pi_order=index%2)
        # Use an actual alternating ring, independent of dictionary edge order.
        double={frozenset(e) for e in [(0,1),(2,3),(4,5)]}
        for edge in graph.edges:
            pi=int(frozenset(edge) in double);graph.edges[edge].update(kekule_order=1+pi,pi_order=pi)
        expected=MorganTokenizer().tokens_graph(graph,3)
        for attrs in graph.edges.values():attrs.update(kekule_order=3-attrs['kekule_order'],pi_order=1-attrs['pi_order'])
        self.assertEqual(MorganTokenizer().tokens_graph(graph,3),expected)
    def test_hash_collision_does_not_merge_distinct_physical_supports(self):
        graph=molecule(4,[(0,1),(2,3)])
        with patch('synrfp.tokenizers.morgan._h64',return_value=1):
            tokens,ledger=MorganTokenizer().tokenize_with_provenance(graph,3)
        self.assertEqual(tokens,Counter({1:6}));self.assertEqual(sum(x.radius==1 for x in ledger),2)
    def test_periodic_lookup_and_unknowns_do_not_become_carbon(self):
        self.assertEqual(len(_ATOMIC_NUMBERS),118);self.assertEqual(_ATOMIC_NUMBERS['Og'],118)
        tokenizer=MorganTokenizer();unknown=Molecule.from_dicts({0:{}},{});carbon=Molecule.from_dicts({0:{'element':'C'}},{})
        self.assertNotEqual(tokenizer.tokens_graph(unknown,0),tokenizer.tokens_graph(carbon,0));self.assertEqual(tokenizer.atom_invariants(unknown)[0][2],('unknown_element',None))
        fake=Molecule.from_dicts({0:{'element':'Qq'}},{})
        self.assertNotEqual(tokenizer.tokens_graph(fake,0),tokenizer.tokens_graph(unknown,0))
    def test_same_support_uses_minimum_identifier_not_first_atom(self):
        graph=molecule(2,[(0,1)],{0:'C',1:'O'})
        def fake_hash(value):
            if value[0]=='ecfp0':return 10 if value[2][2]==('Z',6) else 20
            return 900 if value[3]==10 else 100
        with patch('synrfp.tokenizers.morgan._h64',side_effect=fake_hash):
            _,ledger=MorganTokenizer().tokenize_with_provenance(graph,1)
        emitted=[x for x in ledger if x.radius];self.assertEqual(len(emitted),1);self.assertEqual((emitted[0].center,emitted[0].token),(1,100))
    def test_invalid_graph_and_numeric_states_refuse(self):
        for value in [float('nan'),-1,1.5]:
            graph=Molecule.from_dicts({0:dict(element='C',hcount=value)}, {})
            with self.assertRaises(ValueError):MorganTokenizer().tokens_graph(graph,0)
        with self.assertRaises(ValueError):MorganTokenizer().tokens_graph(Molecule.from_dicts({0:{}},{(0,0):{'order':1}}),1)
    def test_production_import_and_graph_tokenization_need_no_rdkit(self):
        script="""import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='rdkit' or name.startswith('rdkit.'):raise AssertionError('unexpected RDKit runtime dependency')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from synrfp.graph.molecule import Molecule
from synrfp.tokenizers.morgan import MorganTokenizer
g=Molecule.from_dicts({0:{'element':'C'},1:{'element':'O'}},{(0,1):{'order':1}})
assert sum(MorganTokenizer().tokens_graph(g,3).values())==3
"""
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


class RDKitBoundedOracle(unittest.TestCase):
    def test_declared_fixture_radius_supports_and_occurrence_counts(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:self.skipTest('RDKit is an optional audit oracle, not a tokenizer dependency')
        for smiles in ['C','CC','CCC','CCCC','CC.CC','C1CC1','C1CCC1','c1ccccc1','CC(C)O']:
            mol=Chem.MolFromSmiles(smiles);graph=Molecule.from_dicts({a.GetIdx():dict(element=a.GetSymbol(),charge=a.GetFormalCharge(),hcount=a.GetTotalNumHs(),aromatic=a.GetIsAromatic(),isotope=a.GetIsotope()) for a in mol.GetAtoms()}, {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):dict(order=b.GetBondTypeAsDouble(),aromatic=b.GetIsAromatic()) for b in mol.GetBonds()})
            for radius in range(4):
                generator=rdFingerprintGenerator.GetMorganGenerator(radius=radius,includeChirality=False,includeRedundantEnvironments=False)
                additional=rdFingerprintGenerator.AdditionalOutput();additional.CollectBitInfoMap();fp=generator.GetSparseCountFingerprint(mol,additionalOutput=additional)
                expected=Counter()
                for occurrences in additional.GetBitInfoMap().values():
                    for center,r in occurrences:
                        bonds=frozenset(tuple(sorted((mol.GetBondWithIdx(i).GetBeginAtomIdx(),mol.GetBondWithIdx(i).GetEndAtomIdx()))) for i in Chem.FindAtomEnvironmentOfRadiusN(mol,r,center)) if r else frozenset()
                        expected[r,bonds]+=1
                tokens,ledger=MorganTokenizer().tokenize_with_provenance(graph,radius)
                self.assertEqual(Counter((x.radius,x.bonds) for x in ledger),expected,(smiles,radius))
                self.assertEqual(sum(tokens.values()),sum(fp.GetNonzeroElements().values()),(smiles,radius))

if __name__=='__main__':unittest.main()
