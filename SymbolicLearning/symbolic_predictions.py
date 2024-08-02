"""## Importing the libraries"""
import json
import pandas as pd
from rdflib.plugins.sparql.processor import SPARQLResult
from pandasql import sqldf
from rdflib import Graph, URIRef
import re
import os
import time
from validation import travshacl
from transformation import transform
from alive_progress import alive_bar

"""## Loading RDF Graph"""
def load_graph(file):
    g1 = Graph()
    with open(file, "r", encoding="utf-8") as rdf_file:
        lines = rdf_file.readlines()
        for line_number, line in enumerate(lines, start=1):
            try:
                g1.parse(data=line, format="nt")
                # Successfully parsed this line
            except Exception as e:
                print(f"Error parsing line {line_number}: {e}")
    return g1

def sparql_results_to_df(results: SPARQLResult) -> pd.DataFrame:
    return pd.DataFrame(
        data=([None if x is None else x.toPython() for x in row] for row in results),
        columns=[str(x) for x in results.vars],
    )

"""### Generating Symbolic Learning Predictions and Enriching KG"""
def rdflib_query(rule_df, prefix_query, rdf_data,head_val, predictions_folder):
    global query, body_str1, body_str2, qres_df, new_result_df, new_res_df
    result_df = pd.DataFrame()
    new_result_df = pd.DataFrame()
    for idx, item in rule_df.iterrows():
        sub_dataframe = pd.DataFrame([item])
        for i, val in sub_dataframe.iterrows():
            fun_var = val['Functional_variable']
            body = val['Body']
            head = val['Head']
            conf = val['Std_Confidence']

            # Split the input string into individual words
            words = body.split()
            head_split = head.split()
            # Define the prefix
            prefix = 'ex:'
            # Define the regular expression pattern to match words without special characters like "?"
            pattern = re.compile(r'^\w+$')
            # Iterate through the list and modify the elements accordingly
            modified_list = [prefix + item if pattern.match(item) else item for item in words]
            modified_head = [prefix + item if pattern.match(item) else item for item in head_split]
            new_head = ' '.join(modified_head)
            head1 = ' '.join(modified_head[:2] + ['?b'])
            new_head_val = [item if pattern.match(item) else item for item in head_split]
            head2 = new_head_val[2]

            # Split the list into two parts after every three elements
            split_index = 3
            part1 = modified_list[:split_index]
            part2 = modified_list[split_index:]

            # Join the parts into strings
            string1 = ' '.join(part1)
            string2 = ' '.join(part2)

            # Print the strings if they are non-empty
            if string1:
                body_str1 = string1 + "."
            else:
                body_str1 = ""
            if string2:
                body_str2 = string2 + "."
            else:
                body_str2 = ""

            if fun_var == '?a':
                query = f"""
                            PREFIX ex: <{prefix_query}>
                            SELECT DISTINCT ?a WHERE{{
                            {body_str1}
                            {body_str2}
                            FILTER(!EXISTS {{{new_head}}})
                            }}"""
                # print(query)
                file_triple = load_graph(file=rdf_data)
                qres = file_triple.query(query)
                qres_df = pd.DataFrame(qres, columns=qres.vars)
                qres_df['object'] = head2

            else:
                h = new_head.replace("?a", "?a1")
                query = f"""
                               PREFIX ex: <{prefix_query}>
                               SELECT DISTINCT ?a WHERE{{
                               {body_str1}
                               {body_str2}
                               {h} .
                               FILTER(?a1 != ?a).
                               FILTER(!EXISTS {{{new_head}}})
                               }}"""
                # print(query)
                file_triple = load_graph(file=rdf_data)
                qres = file_triple.query(query)
                qres_df = pd.DataFrame(qres, columns=qres.vars)
                qres_df['object'] = head2

        result_df = pd.concat([result_df, qres_df], ignore_index=True)

    result_df = result_df.replace(prefix_query, '', regex=True)
    result_df.insert(loc=1, column='predicate', value=head_val)

    result_df.rename(columns={result_df.columns[0]: 'subject'}, inplace=True)

    # Initialize an empty graph
    g = Graph()
    g.parse(rdf_data, format='nt')
    # Iterate over the DataFrame and add triples to the graph
    for index, row in result_df.iterrows():
        subject = URIRef(prefix_query + row['subject'])
        predicate = URIRef(prefix_query + row['predicate'])
        object = URIRef(prefix_query + row['object'])
        g.add((subject, predicate, object))

    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    # Serialize the graph to N-Triples format and save to a new file
    enrichedKG = '../KG/EnrichedKG/Enriched_KG.nt'
    g.serialize(destination=enrichedKG, format='nt')
    print(f"Enriched knowledge graph saved to {enrichedKG}")
    result_df.to_csv(predictions_folder + f'/{head_val}.tsv', sep='\t', index=False, header=None)

    return result_df, g

def readRules(file, prefix, rdf_data, predictions_folder):
    rules = pd.read_csv(file)
    q1 = f"""SELECT DISTINCT Head, COUNT(*) AS num FROM rules GROUP BY Head ORDER BY num DESC"""
    head_df = sqldf(q1, locals())
    for i, val in head_df.iterrows():
        head = val['Head']
        head_val = head.split()[1]
        q2 = f"""SELECT * FROM rules WHERE Head LIKE '%{head}%' ORDER BY Std_Confidence DESC"""
        rule = sqldf(q2, locals())
        result, enrichedKG = rdflib_query(rule, prefix, rdf_data, head_val, predictions_folder)
    return result, enrichedKG


def initialize(input_config):
    with open(input_config, "r") as input_file_descriptor:
        input_data = json.load(input_file_descriptor)
    prefix = input_data['prefix']
    path = '../KG/'+input_data['KG']
    rules = './Rules/'+ input_data['rules_file']
    rdf = path +"/"+ input_data['rdf_file']
    predictions_folder = './Predictions/'
    constraints = './'+ input_data['constraints_folder']

    return prefix, rules, rdf, path, predictions_folder, constraints

if __name__ == '__main__':
    start_time = time.time()
    input_config = 'input.json'

    # Reading input.json file to collect input configuration for executing symbolic learning
    prefix, rulesfile, rdf_data, path, predictions_folder, constraints = initialize(input_config)

    # Processing rules and generating predictions based on PCA heuristics
    rule_df, enrichedKG = readRules(rulesfile, prefix, rdf_data, predictions_folder)

    val_results = travshacl(enrichedKG, constraints)

    transform(enrichedKG)

    end_time = time.time()
    execution_time = end_time - start_time

    print('Elaspsed time', execution_time)

