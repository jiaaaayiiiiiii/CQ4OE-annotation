#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ontology_primitive_core_terms.py
=================================
Functions:
1. Automatically parse BNode restrictions in the ontology
2. Convert restrictions into explicit edges
3. Compute in-degree / out-degree / total-degree for each term
4. Identify primitive classes (classes without necessary & sufficient conditions)
5. Output:
   - term_degree_metrics.csv  : degree metrics for all nodes
   - extracted_edges.csv      : all extracted edges
   - primitive_terms.csv      : ALL primitive classes with degree information
   - core_terms.csv           : top N% terms from the ENTIRE ontology (all nodes)
"""

import os
from collections import defaultdict

import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL, URIRef, BNode, Literal


# ========== Configuration ==========
TOP_PERCENTAGE = 0.1          # Top N% of ALL terms to output as core terms
# ===================================


def guess_format(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".owl", ".rdf", ".xml"]:
        return "xml"
    if ext == ".ttl":
        return "turtle"
    if ext == ".nt":
        return "nt"
    if ext == ".jsonld":
        return "json-ld"
    return None


def short_name(term):
    if term is None:
        return ""
    if isinstance(term, BNode):
        return f"_:{str(term)}"
    if isinstance(term, URIRef):
        uri = str(term)
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.rstrip("/").split("/")[-1]
    if isinstance(term, Literal):
        return str(term)
    return str(term)


def load_graph(file_path: str) -> Graph:
    g = Graph()
    fmt = guess_format(file_path)
    g.parse(file_path, format=fmt)
    return g


def extract_classes(g: Graph):
    classes = set()
    for c in g.subjects(RDF.type, OWL.Class):
        if isinstance(c, URIRef):
            classes.add(c)
    for c in g.subjects(RDF.type, RDFS.Class):
        if isinstance(c, URIRef):
            classes.add(c)
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, URIRef):
            classes.add(s)
        if isinstance(o, URIRef):
            classes.add(o)
    return classes


def extract_properties(g: Graph):
    props = set()
    property_types = [
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.FunctionalProperty,
        OWL.TransitiveProperty,
        OWL.SymmetricProperty,
        OWL.AsymmetricProperty,
        OWL.InverseFunctionalProperty,
    ]
    for ptype in property_types:
        for p in g.subjects(RDF.type, ptype):
            if isinstance(p, URIRef):
                props.add(p)
    for s, o in g.subject_objects(RDFS.subPropertyOf):
        if isinstance(s, URIRef):
            props.add(s)
        if isinstance(o, URIRef):
            props.add(o)
    for p in g.subjects(RDFS.domain, None):
        if isinstance(p, URIRef):
            props.add(p)
    for p in g.subjects(RDFS.range, None):
        if isinstance(p, URIRef):
            props.add(p)
    # Include properties appearing in restrictions
    for rnode in g.subjects(RDF.type, OWL.Restriction):
        for p in g.objects(rnode, OWL.onProperty):
            if isinstance(p, URIRef):
                props.add(p)
    return props


def add_edge(edges, source, target, edge_type):
    if source is None or target is None:
        return
    edges.add((source, target, edge_type))


def build_explicit_edges(g: Graph, classes: set, props: set):
    edges = set()
    # subclass
    for child, parent in g.subject_objects(RDFS.subClassOf):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            if child in classes and parent in classes:
                add_edge(edges, child, parent, "subClassOf")
    # domain/range
    for p in props:
        for d in g.objects(p, RDFS.domain):
            if isinstance(d, URIRef):
                add_edge(edges, d, p, "domain_to_property")
        for r in g.objects(p, RDFS.range):
            if isinstance(r, URIRef):
                add_edge(edges, p, r, "property_to_range")
    # subPropertyOf
    for child, parent in g.subject_objects(RDFS.subPropertyOf):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            if child in props and parent in props:
                add_edge(edges, child, parent, "subPropertyOf")
    return edges


def build_restriction_edges(g: Graph):
    edges = set()
    filler_predicates = [
        OWL.someValuesFrom,
        OWL.allValuesFrom,
        OWL.onClass,
        OWL.hasValue,
    ]
    for class_node, restriction_node in g.subject_objects(RDFS.subClassOf):
        if not isinstance(class_node, URIRef):
            continue
        if not isinstance(restriction_node, BNode):
            continue
        if (restriction_node, RDF.type, OWL.Restriction) not in g:
            continue
        on_properties = list(g.objects(restriction_node, OWL.onProperty))
        if not on_properties:
            continue
        for prop in on_properties:
            if not isinstance(prop, URIRef):
                continue
            add_edge(edges, class_node, prop, "restriction_onProperty")
            for pred in filler_predicates:
                for filler in g.objects(restriction_node, pred):
                    if isinstance(filler, URIRef):
                        add_edge(edges, prop, filler, short_name(pred))
                    elif isinstance(filler, Literal):
                        add_edge(edges, prop, filler, short_name(pred))
    return edges


def collect_nodes_from_edges(edges):
    nodes = set()
    for s, o, _ in edges:
        nodes.add(s)
        nodes.add(o)
    return nodes


def compute_degrees(nodes, edges):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for s, o, _ in edges:
        out_degree[s] += 1
        in_degree[o] += 1
    rows = []
    for node in nodes:
        indeg = in_degree.get(node, 0)
        outdeg = out_degree.get(node, 0)
        rows.append({
            "term_uri": str(node),
            "term_name": short_name(node),
            "in_degree": indeg,
            "out_degree": outdeg,
            "total_degree": indeg + outdeg,
        })
    return rows, in_degree, out_degree


def determine_term_type(node, classes, props):
    if isinstance(node, Literal):
        return "Literal"
    if node in classes:
        return "Class"
    if node in props:
        return "Property"
    if isinstance(node, BNode):
        return "BNode"
    return "Other"


def build_edge_dataframe(edges, classes, props):
    rows = []
    for s, o, etype in sorted(edges, key=lambda x: (short_name(x[0]).lower(), short_name(x[1]).lower(), x[2])):
        rows.append({
            "source_uri": str(s),
            "source_name": short_name(s),
            "source_type": determine_term_type(s, classes, props),
            "target_uri": str(o),
            "target_name": short_name(o),
            "target_type": determine_term_type(o, classes, props),
            "edge_type": etype,
        })
    return pd.DataFrame(rows)


def is_primitive_class(g: Graph, class_uri):
    """Determine if a class is primitive (no necessary & sufficient conditions)."""
    if not isinstance(class_uri, URIRef):
        return False
    # Check for equivalentClass (other than itself)
    for eq in g.objects(class_uri, OWL.equivalentClass):
        if eq != class_uri:
            return False
    # Check if the class itself uses complex class constructors
    if (class_uri, OWL.intersectionOf, None) in g:
        return False
    if (class_uri, OWL.unionOf, None) in g:
        return False
    if (class_uri, OWL.oneOf, None) in g:
        return False
    return True


def main(input_ontology_path: str, output_dir: str = "ontology_primitive_core_output"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading ontology: {input_ontology_path}")
    g = load_graph(input_ontology_path)
    print(f"[INFO] Triple count: {len(g)}")

    classes = extract_classes(g)
    props = extract_properties(g)
    print(f"[INFO] Classes: {len(classes)}")
    print(f"[INFO] Properties: {len(props)}")

    explicit_edges = build_explicit_edges(g, classes, props)
    restriction_edges = build_restriction_edges(g)
    all_edges = explicit_edges | restriction_edges
    print(f"[INFO] Total edges: {len(all_edges)}")

    all_nodes = collect_nodes_from_edges(all_edges)
    degree_rows, in_degree_dict, out_degree_dict = compute_degrees(all_nodes, all_edges)

    # Add term_type to degree rows
    for row in degree_rows:
        uri = row["term_uri"]
        node = None
        if uri.startswith("http://") or uri.startswith("https://"):
            node = URIRef(uri)
        else:
            node = Literal(uri)
        row["term_type"] = determine_term_type(node, classes, props)

    df_degree = pd.DataFrame(degree_rows)
    df_degree = df_degree.sort_values(
        by=["total_degree", "in_degree", "out_degree", "term_name"],
        ascending=[False, False, False, True]
    )

    # ========== 1. core_terms.csv: top N% of ALL terms (any type) ==========
    total_terms = len(df_degree)
    top_n_all = max(1, int(total_terms * TOP_PERCENTAGE))
    core_df = df_degree.head(top_n_all).copy()
    # Optionally filter to keep only Class and Property (remove Literal/BNode/Other)
    # core_df = core_df[core_df["term_type"].isin(["Class", "Property"])].copy()
    # If you want to keep all types, leave as is.
    core_file = os.path.join(output_dir, "core_terms.csv")
    core_df.to_csv(core_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] Core terms (top {TOP_PERCENTAGE*100}% of all {total_terms} terms): {core_file}")

    # ========== 2. primitive_terms.csv: all primitive classes with degree info ==========
    primitive_classes = {c for c in classes if is_primitive_class(g, c)}
    print(f"[INFO] Primitive classes: {len(primitive_classes)}")

    primitive_rows = []
    for cls in primitive_classes:
        uri_str = str(cls)
        name = short_name(cls)
        indeg = in_degree_dict.get(cls, 0)
        outdeg = out_degree_dict.get(cls, 0)
        primitive_rows.append({
            "term_uri": uri_str,
            "term_name": name,
            "in_degree": indeg,
            "out_degree": outdeg,
            "total_degree": indeg + outdeg,
            "term_type": "Class"
        })
    df_primitive = pd.DataFrame(primitive_rows)
    df_primitive = df_primitive.sort_values(
        by=["total_degree", "in_degree", "out_degree", "term_name"],
        ascending=[False, False, False, True]
    )
    primitive_file = os.path.join(output_dir, "primitive_terms.csv")
    df_primitive.to_csv(primitive_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] All primitive classes: {primitive_file}")

    # ========== 3. Other outputs ==========
    degree_file = os.path.join(output_dir, "term_degree_metrics.csv")
    edge_file = os.path.join(output_dir, "extracted_edges.csv")
    df_degree.to_csv(degree_file, index=False, encoding="utf-8-sig")
    build_edge_dataframe(all_edges, classes, props).to_csv(edge_file, index=False, encoding="utf-8-sig")

    print(f"[DONE] Full degree file: {degree_file}")
    print(f"[DONE] Edge file: {edge_file}")


if __name__ == "__main__":
    # Modify these paths as needed
    input_ontology_path = "/Users/ljymacbook/LLMs_Ontology_experiments/Stuff/stuff.owl"
    output_dir = "/Users/ljymacbook/LLMs_Ontology_experiments/Stuff/ontology_primitive_core_output"
    main(input_ontology_path, output_dir)