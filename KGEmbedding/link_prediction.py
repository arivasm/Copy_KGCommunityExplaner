import argparse
import pandas as pd
from ranks import  tail_prediction, head_prediction, load_dataset
import torch

def main(args):
    embedding_model = torch.load(args.results_path + args.model_name + '/trained_model.pkl', map_location=torch.device('cpu'))
    training_triples, triple_data1, entity_label, relation_label = load_dataset(args.results_path +'/train')
    pred = tail_prediction(embedding_model, args.head, args.relation, training_triples)
    pred.to_csv(args.results_path + args.model_name + '/prediction_result.csv', index=False)
    
    return pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--results_path', type=str, required=True, help='path to stored results of trained kge model')
    parser.add_argument('--model_name', type=str, required=True, help='name of kge model')
    parser.add_argument('--head', type=str, required=True, help='head entity')
    parser.add_argument('--relation', type=str, required=True, help='relation')

    args = parser.parse_args()
    main(args)