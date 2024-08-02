import pandas as pd
import json
import numpy as np

agg_dict = {}

def average_of_two_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    combined = [a + b for a, b in zip(list1, list2)]
    return [sum / 2 for sum in combined]

def load_vectors(data, path_to_embedding_model):
    average_dict = {}
    # Load embeddings from separate JSON files
    with open(f'{path_to_embedding_model}' + 'entity_representation.json', 'r') as f:
        entity_embeddings = json.load(f)

    with open(f'{path_to_embedding_model}' + 'relation_representation.json', 'r') as f:
        relation_embeddings = json.load(f)

    # Function to get embedding vector
    def get_embedding(item, item_type):
        if item_type == 'entity':
            return entity_embeddings.get(item, None)
        elif item_type == 'relation':
            return relation_embeddings.get(item, None)
        else:
            return None

    for ind, row in data.iterrows():
        subject = row['subject']
        sub_vec = get_embedding(row['subject'], 'entity')
        rel_vec = get_embedding(row['predicate'], 'relation')
        en_vec = get_embedding(row['object'], 'entity')
        vec_avg = (np.array(rel_vec) + np.array(en_vec))/2
        # Stack the arrays along a new axis
        stacked_arrays = np.stack(vec_avg, axis=0)
        # Compute the final average with the single array
        final_average = np.mean([stacked_arrays, sub_vec], axis=0).flatten()
        final_average_list = final_average.tolist()
        average_dict[subject] = final_average_list
    return average_dict

def aggregate_ego_network(ego_network, path_to_embedding_model):
    dict = load_vectors(ego_network, path_to_embedding_model)
    return dict

