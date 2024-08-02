from pykeen.pipeline import pipeline, plot_losses
import numpy as np
import pandas as pd
from pykeen import predict
from pykeen.triples import TriplesFactory
from matplotlib import pyplot as plt
from typing import List
import pykeen.nn
import torch
import json
import os.path
import logging

def save_vectors(entity_representation_tensor,relation_embedding_tensor, entity_labels, relation_labels, m, path):
    entity_representation_tensor = entity_representation_tensor.cpu().detach().numpy()
    relation_embedding_tensor = relation_embedding_tensor.cpu().detach().numpy()
    vectors_path = path + f'/{m}/vectors'
    if not os.path.exists(vectors_path):
        os.makedirs(vectors_path)
    entity_dict = {label: vector.tolist() for label, vector in zip(entity_labels, entity_representation_tensor)}
    relation_dict = {label: vector.tolist() for label, vector in zip(relation_labels, relation_embedding_tensor)}

    with open(os.path.join(vectors_path, "entity_representation.json"), 'w') as entity_file:
        json.dump(entity_dict, entity_file)

    with open(os.path.join(vectors_path, "relation_representation.json"), 'w') as relation_file:
        json.dump(relation_dict, relation_file)

# to store the entity and relation representation
def get_learned_embeddings(model):
    entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.RepresentationModule'] = model.relation_representations

    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()

    return entity_embedding_tensor, relation_embedding_tensor

def load_dataset(name):
    triple_data = open(name, encoding='utf-8').read().strip()
    data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
    tf_data = TriplesFactory.from_labeled_triples(triples=data)
    entity_label =tf_data.entity_to_id.keys()
    relation_label = tf_data.relation_to_id.keys()
    return tf_data, triple_data, entity_label, relation_label

def create_model(tf_training, tf_testing, embedding, n_epoch, path):
    results = pipeline(
        training=tf_training,
        testing=tf_testing,
        model=embedding,
        training_loop='sLCWA',
        model_kwargs=dict(embedding_dim=50),
        # Training configuration
        training_kwargs=dict(
            num_epochs=n_epoch,
            batch_size = 896,
            use_tqdm_batch=False,
        ),
        negative_sampler_kwargs=dict(
            filtered=True,
        ),
        optimizer= 'Adam',
        # Runtime configuration
        random_seed=1235,
    )
    model = results.model
    results.save_to_directory(path + embedding)
    return model, results

def plotting(results,m, path):
        plot_losses(results)
        plt.savefig(path + m + "/loss_plot.png", dpi=300)

def save_triples_as_json(training_triples, testing_triples, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    training_dict = training_triples.to_dict(orient='records')
    testing_dict = testing_triples.to_dict(orient='records')

    training_file_path = os.path.join(directory, 'training_triples.json')
    testing_file_path = os.path.join(directory, 'testing_triples.json')

    with open(training_file_path, 'w') as training_file:
        json.dump(training_dict, training_file, indent=4)

    with open(testing_file_path, 'w') as testing_file:
        json.dump(testing_dict, testing_file, indent=4)

def dataframe_embedding_donors(entity_embedding_tensor, training, name, path):
    kg = pd.read_csv(name, delimiter="\t", header=None)
    kg.columns = ['s', 'p', 'o']
    entity = list(kg.loc[kg.p == 'hasGender'].s.unique())
    df = pd.DataFrame(entity_embedding_tensor.cpu().detach().numpy())
    df['ClinicalRecord'] = list(training.entity_to_id)
    new_df = df.loc[df.ClinicalRecord.isin(list(entity))]
    return new_df

# to predict the tail entity
def tail_prediction(model, head, relation, training):
    pred = predict.predict_target(model=model, head=head, relation=relation, triples_factory=training).df
    return pred

# to predict the head entity
def head_prediction(model, relation, tail, training, label):
      pred = predict.predict_target(model=model, relation= relation, tail= tail, triples_factory=training).df
      return pred
