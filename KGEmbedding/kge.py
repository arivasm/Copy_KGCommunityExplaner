import argparse
import pandas as pd
from ranks import load_dataset, create_model, get_learned_embeddings, plotting, dataframe_embedding_donors

def main(args):
    # Load the dataset
    triples, triple_data, entity_label, relation_label = load_dataset(args.dataset_path)

    # Split the dataset
    training, testing, validation = triples.split(ratios=[0.8,0.1,0.1], random_state=1234)
    print(training)
    print(testing)
    print(validation)

    # Save training triples
    training_triples = pd.DataFrame(training.triples)
    training_triples.to_csv(f'{args.output_dir}/train', index=False, header=False, sep='\t')

    # Save testing triples
    testing_triples = pd.DataFrame(testing.triples)
    testing_triples.to_csv(f'{args.output_dir}/test', index=False, header=False, sep='\t')

    # Save validation triples
    valid_triples = pd.DataFrame(validation.triples)
    valid_triples.to_csv(f'{args.output_dir}/valid', index=False, header=False, sep='\t')

    for model_name in args.models:
        model, results = create_model(tf_training=training, tf_testing=testing, embedding=model_name, n_epoch=100, path=args.results_path)
        entity_representation_tensor, relation_embedding_tensor = get_learned_embeddings(model)
        plotting(results, model_name, args.results_path)
        patient = dataframe_embedding_donors(entity_representation_tensor, training, args.dataset_path, args.results_path)
        patient.to_csv(f'{args.results_path}/{model_name}/embedding_donors.csv', index=None)
    return model, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    parser.add_argument('--results_path', type=str, required=True, help='Path to store the results')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to use')

    args = parser.parse_args()
    main(args)
