"""Maintenance knowledge graph (shared schema, local instances). Phase 4 - to be implemented.

Planned schema (ontology):
  Machine -HAS_COMPONENT-> Component -MONITORED_BY-> Sensor
  Component -EXHIBITS-> FailureMode -INDICATED_BY-> Symptom (sensor trend / error code)
  FailureMode -RESOLVED_BY-> MaintenanceAction
"""
import networkx as nx

NODE_TYPES = ["Machine", "Component", "Sensor", "ErrorCode", "FailureMode", "MaintenanceAction"]
EDGE_TYPES = ["HAS_COMPONENT", "MONITORED_BY", "EXHIBITS", "INDICATED_BY", "RESOLVED_BY"]


def new_graph():
    return nx.MultiDiGraph()


def add_fact(g, head, head_type, rel, tail, tail_type):
    assert head_type in NODE_TYPES and tail_type in NODE_TYPES and rel in EDGE_TYPES
    g.add_node(head, type=head_type)
    g.add_node(tail, type=tail_type)
    g.add_edge(head, tail, rel=rel)


def retrieve_facts(g, entity, hops=2):
    """Return triples within `hops` of an entity, as text lines for LLM grounding."""
    nodes = nx.single_source_shortest_path_length(g.to_undirected(), entity, cutoff=hops)
    return [f"{u} --{d['rel']}--> {v}" for u, v, d in g.edges(data=True) if u in nodes and v in nodes]
