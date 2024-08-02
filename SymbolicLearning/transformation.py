import os
import re
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import SH, RDF

# Function to read and validate file content
def read_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"File is empty: {file_path}")
    return content

def read_file_prefix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"File is empty: {file_path}")
        if '@prefix :' not in content:
            content = '@prefix : <http://example.org/> .\n' + content
    return content

def read_log_file(log_file):
    valid_patients = []
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"File not found: {log_file}")
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"File is empty: {log_file}")
        matches = re.findall(r'<(.*?)>\((.*?)\)', content)
        valid_patients = [(URIRef(shape), URIRef(patient)) for shape, patient in matches]
    return valid_patients


def transform_shacl_shapes(shacl_shapes_file, shacl_report_file):
    # Load SHACL shapes and validation report into RDF graphs
    shapes_graph = Graph()
    shapes_content = read_file(shacl_shapes_file)
    shapes_graph.parse(data=shapes_content, format='turtle')

    report_graph = Graph()
    report_content = read_file_prefix(shacl_report_file)
    report_graph.parse(data=report_content, format='turtle')

    # Define namespaces
    LUNG_CANCER_S = Namespace("http://example.org/lungCancer/shapes/")
    LUNG_CANCER_E = Namespace("http://example.org/lungCancer/entity/")
    LUNG_CANCER_V = Namespace("http://example.org/lungCancer/vocab/")

    # Extract violations from the SHACL validation report
    violations = set()
    for result in report_graph.objects(None, SH.result):
        focus_node = report_graph.value(result, SH.focusNode)
        source_shape = report_graph.value(result, SH.sourceShape)
        violations.add((str(focus_node), str(source_shape)))


    # Extract patterns from SHACL shapes
    patterns = {}
    for shape in shapes_graph.subjects(RDF.type, SH.NodeShape):
        for sparql in shapes_graph.objects(shape, SH.sparql):
            query_text = shapes_graph.value(sparql, SH.select)
            if query_text:
                # Extract patterns within FILTER EXISTS {}
                filter_patterns = re.findall(r'FILTER EXISTS \{(.*?)\}', str(query_text), re.DOTALL)
                patterns[str(shape)] = filter_patterns

    return violations, patterns


# Function to update the original KG based on violations and patterns
def update_kg(kg_graph, violations, patterns):
    for focus_node, source_shape in violations:
        subject = URIRef(focus_node)
        if source_shape in patterns:
            for pattern in patterns[source_shape]:
                # Extract predicate and object from the pattern
                match = re.search(r'<(.*?)>\s*<(.*?)>', pattern)
                if match:
                    pred = URIRef(match.group(1))
                    obj = URIRef(match.group(2))
                    # Find and modify the triples in the original KG
                    for s, p, o in kg_graph.triples((subject, None, None)):
                        if (p == pred) and (o == obj):
                            new_pred = URIRef(f"{str(pred)}_No{str(obj).split('/')[-1]}")
                            new_obj = obj.rsplit('/', 1)
                            new_obj = URIRef(f"{str(new_obj[0])+'/No'+str(new_obj[1])}")
                            kg_graph.remove((s, p, o))
                            kg_graph.add((s, new_pred, new_obj))
                        if (p == pred):
                            new_pred = URIRef(f"{str(pred)}_{str(o).split('/')[-1]}")
                            new_obj = URIRef(o)
                            kg_graph.remove((s, p, o))
                            kg_graph.add((s, new_pred, new_obj))
    return kg_graph

# Function to update the original KG based on valid patients
def update_kg_for_valid_patients(kg_graph, valid_patients, patterns):
    for source_shape, patient in valid_patients:
        subject = patient
        if str(source_shape) in patterns:
            for pattern in patterns[str(source_shape)]:
                # Extract predicate and object from the pattern
                match = re.search(r'<(.*?)>\s*<(.*?)>', pattern)
                if match:
                    pred = URIRef(match.group(1))
                    obj = URIRef(match.group(2))
                    # Find and modify the triples in the original KG
                    for s, p, o in kg_graph.triples((subject, None, None)):
                        if (p == pred):
                            new_pred = URIRef(f"{str(p)}_{str(o).split('/')[-1]}")
                            new_obj = URIRef(o)
                            kg_graph.remove((s, p, o))
                            kg_graph.add((s, new_pred, new_obj))

    return kg_graph

def transform(enrichedKG):
    # File paths
    shacl_shapes_file = 'Constraints/LungCancer-shapes.ttl'
    shacl_violation_report = 'Constraints/result/validationReport.ttl'
    output_kg_file = '../KG/TransformedKG/TransformedKG.nt'
    shacl_valid_report = 'Constraints/result/targets_valid.log'

    violations, patterns = transform_shacl_shapes(shacl_shapes_file, shacl_violation_report)

    transformedKG_invalid = update_kg(enrichedKG, violations, patterns)

    # Read valid patients from the log file
    valid_patients = read_log_file(shacl_valid_report)

    # Update the KG for valid patients
    transformedKG = update_kg_for_valid_patients(transformedKG_invalid, valid_patients, patterns)

    # Save the updated KG to a file
    transformedKG.serialize(destination=output_kg_file, format='nt')
    print(f"Transformed knowledge graph saved to {output_kg_file}")
