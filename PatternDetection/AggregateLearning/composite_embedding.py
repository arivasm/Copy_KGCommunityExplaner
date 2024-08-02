from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import sys
from vec_agg import aggregate_ego_network

sparql = SPARQLWrapper("https://labs.tib.eu/sdm/LungCancer/sparql")
entity_type = 'http://example.org/lungCancer/entity/Patient'


def adding_prefix(df_g):
    df_g.replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', '', regex=True, inplace=True)
    df_g.replace('http://example.org/lungCancer/entity/', '', regex=True, inplace=True)
    return df_g


def retrieve_entity(entity_type, sparql):
    q_entity = """
    prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT DISTINCT ?ego_entity
    WHERE {
      ?ego_entity a <"""+entity_type+"""> .

    }
    """
    sparql.setQuery(q_entity)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

def extract_ego_network(entity, sparql):
    #     === Query to retrieve the ego network
    q_ego_network = """
    SELECT DISTINCT ?subject ?predicate ?object
    WHERE {
      {
        <""" + entity + """> ?predicate ?object .
        BIND(<""" + entity + """> AS ?subject)
      }
      OPTIONAL {
        ?subject ?predicate <""" + entity + """> .
        BIND(<""" + entity + """> AS ?object)
      }
    }
    """
    sparql.setQuery(q_ego_network)
    sparql.setReturnFormat(JSON)
    triples = sparql.query().convert()
    return triples


def create_dataframe_from_nested_dict(aggregate_vector):
    rows = []
    # Populate rows with the data
    for model, entities_list in aggregate_vector.items():
        for entities in entities_list:
            for entity, values in entities.items():
                row = values + [entity, model]  # Concatenate the list values with entity and model
                rows.append(row)

    num_cols = len(rows[0]) - 2
    # Generate column names dynamically
    col_names = list(range(num_cols)) + ['Entity', 'Model']
    aggregate_vector = pd.DataFrame(rows, columns=col_names)
    return aggregate_vector


def composite_embedding(entity_type, endpoint, model_list):
    sparql = SPARQLWrapper(endpoint)
    aggregate_vector = {model: [] for model in model_list}  # Initialize a dictionary with empty lists for each model
    # === Retrieve entities of type T ===
    results = retrieve_entity(entity_type, sparql)
    for r in results['results']['bindings']:
        ego_network = []
        entity = r['ego_entity']['value']
        # === Extract ego network of 'entity' ===
        triples = extract_ego_network(entity, sparql)
        for p in triples['results']['bindings']:
            row = {'subject': p['subject']['value'], 'predicate': p['predicate']['value'],
                   'object': p['object']['value']}
            ego_network.append(row)

        # == Data frame of ego network. Output: four columns, the last one is the ego entity.
        ego_network = pd.DataFrame.from_dict(ego_network)
        # ego_network['ego_entity'] = entity
        ego_network = adding_prefix(ego_network)
        ego_network = ego_network.loc[ego_network['predicate']!='type']

        for model in model_list:
            model_path = "../KGEmbedding/OriginalKG/"+model+"/vectors/"
            aggregate = aggregate_ego_network(ego_network, model_path)
            aggregate_vector[model].append(aggregate)  # Append the aggregate to the list for the current model

            # aggregate = {model: aggregate}
            # aggregate_vector.update(aggregate)
    return create_dataframe_from_nested_dict(aggregate_vector)


def main(*args):
    aggregate_vector = composite_embedding(args[0], args[1], ['TransH'])
    aggregate_vector.to_csv('aggregate_vector.csv', index=None)


if __name__ == '__main__':
    main(*sys.argv[1:])

