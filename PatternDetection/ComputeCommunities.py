import pandas as pd
import os, sys, glob
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import subprocess
import Utility
from AggregateLearning import composite_embedding as aggregate


def call_semEP(threshold, cls_addres, file_addres):
    th = "{:.4f}".format(float(threshold))
    # !./semEP-node file_addres + "ClinicalRecord.txt " + file_addres + "matrix_ClinicalRecord.tsv " + str(th)
    command = f"./semEP-node {file_addres}ClinicalRecord.txt {file_addres}matrix_ClinicalRecord.tsv {th}"
    subprocess.run(command, shell=True, check=True)
    pattern = 'ClinicalRecord'
    results_folder = glob.glob("./" + pattern + "-*")
    onlyfiles = [os.path.join(results_folder[0], f) for f in listdir(results_folder[0]) if 
                 isfile(join(results_folder[0], f))]
    count = 0
    for filename in onlyfiles:
        key = "cluster-" + str(count) + '.txt'
        copyfile(filename, cls_addres + 'clusters/' + key)
        count += 1
    # dicc_clusters = get_dicc_clusters(onlyfiles)

    for r, d, f in os.walk(results_folder[0]):
        for files in f:
            os.remove(os.path.join(r, files))
        os.removedirs(r)
    return len(onlyfiles)


def METIS_Undirected_MAX_based_similarity_graph(cosine_matrix, cls_address_metis):
    metislines = []
    nodes = {"name": [], "id": []}
    kv = 1
    edges = 0
    for i, row in cosine_matrix.iterrows():
        val = ""
        ix = 1
        ledges = 0
        found = False
        for k in row.keys():
            if i != k and row[k] > 0:
                val += str(ix) + " " + str(int(row[k] * 100000)) + " "
                # Only one edge is counted between two nodes, i.e., (u,v) and (v, u) edges are counted as one
                # Self links are also ignored, Notive ix>kv
                # if ix > kv:
                ledges += 1
                found = True
            ix += 1
        if found:
            # This node is connected
            metislines.append(val.strip())
            edges += ledges
            nodes["name"].append(i)
            nodes['id'].append(str(kv))
        else:
            # disconnected RDF-MTs are given 10^6 value as similarity value
            metislines.append(str(kv) + " 100000")
            edges += 1
            # ---------
            nodes["name"].append(i)
            nodes['id'].append(str(kv))
            print(i)
            print(str(kv))

        kv += 1
    nodes = pd.DataFrame(nodes)
    # print(edges)
    numedges = edges // 2
    # == Save filemetis.graph to execute METIS algorithm ==
    ff = open(cls_address_metis + 'metis.graph', 'w+')
    ff.write(str(cosine_matrix.shape[0]) + " " + str(numedges) + " 001\n")
    met = [m.strip() + "\n" for m in metislines]
    ff.writelines(met)
    ff.close()
    return nodes


def call_metis(num_cls, nodes, cls_address_metis):
    # !sudo docker run -it --rm -v /media/rivas/Data1/Data-mining/KCAP-I40KG-Embeddings/I40KG-Embeddings/result/TransD/metis:/data kemele/metis:5.1.0 gpmetis metis.graph 2
    current_path = os.path.dirname(os.path.realpath(__file__))
    EXE_METIS = "sudo docker run --rm -v "
    DIR_METIS = ":/data kemele/metis:5.1.0 gpmetis"
    cls_addres = cls_address_metis[:-1]
    commd = EXE_METIS + current_path + '/' + cls_addres + DIR_METIS + " metis.graph " + str(num_cls)
    # print(commd)
    # os.system(commd)
    subprocess.run(commd, shell=True, check=True)
    # !{commd}
    parts = open(cls_address_metis + 'metis.graph.part.' + str(num_cls)).readlines()
    parts = [p.strip() for p in parts]
    # == Save each partition standads into a file ==
    i = 0
    partitions = dict((str(k), []) for k in range(num_cls))
    for p in parts:
        name = nodes.iat[i, 0]
        i += 1
        partitions[str(p)].append(name)

    i = 0
    count = 0
    for p in partitions:
        if len(partitions[p]) == 0:
            continue
        count += len(partitions[p])
        f = open(cls_address_metis + 'clusters/cluster-' + str(i) + '.txt', 'w+')
        [f.write(l + '\n') for l in partitions[p]]
        f.close()
        i += 1


def cluster_statistics(df, cls_statistics, num_cls, cls_address):
    for c in range(num_cls):
        try:
            No_Progression = df.loc[df.cluster == c][['Relapse']].value_counts()['No_Progression']
        except KeyError:
            No_Progression = 0
        try:
            Progression = df.loc[df.cluster == c][['Relapse']].value_counts()['Progression']
        except KeyError: #AttributeError
            Progression = 0
        try:
            Relapse = df.loc[df.cluster == c][['Relapse']].value_counts()['Relapse']
        except KeyError:
            Relapse = 0
        try:
            UnKnown = df.loc[df.cluster == c][['Relapse']].value_counts()['UnKnown']
        except KeyError:
            UnKnown = 0
        cls_statistics.at['No_Progression', 'cluster-' + str(c)] = int(No_Progression)  # / 14
        cls_statistics.at['Progression', 'cluster-' + str(c)] = int(Progression)  # / 14
        cls_statistics.at['Relapse', 'cluster-' + str(c)] = int(Relapse)  # / 14
        cls_statistics.at['UnKnown', 'cluster-' + str(c)] = int(UnKnown)  # / 73
    cls_statistics.to_csv(cls_address + 'cls_statistics.csv')



def update_cluster_folder(cls_address):
    if os.path.exists(cls_address + 'clusters/'):
        current_path = os.path.dirname(os.path.realpath(__file__))
        results_folder = glob.glob(current_path + '/' + cls_address + 'cluster*')
        for r, d, f in os.walk(results_folder[0]):
            for files in f:
                os.remove(os.path.join(r, files))
    else:
        # os.makedirs(cls_address)
        os.makedirs(cls_address + 'clusters/')

def get_kg(path_kg):
    kg = pd.read_csv(path_kg, delimiter="\t", header=None)
    kg.columns = ['s', 'p', 'o']
    return kg


def get_target(kg, target, df_donor):
    target_kg = kg.loc[kg.p == target]
    target_kg = target_kg.rename(columns={"s": "Entity", "o": "Relapse"})
    target_kg = target_kg[['Entity', 'Relapse']]
    target_kg.drop_duplicates(subset='Entity', inplace=True)
    target_unknown = df_donor.merge(target_kg, how='outer', on='Entity', indicator=True).loc[
        lambda x: x['_merge'] == 'left_only']
    target_unknown.Relapse = 'UnKnown'
    target_unknown = target_unknown[['Entity', 'Relapse']]
    target_kg = pd.concat([target_kg, target_unknown])
    return target_kg


# Function to extract real parts from complex number strings
def extract_real_part(complex_str):
    complex_num = complex(complex_str.strip("'"))
    return complex_num.real


def run(entity_type, endpoint, kg_name, target_predicate, model_list, threshold):
    kg = get_kg('../KG/'+kg_name+'/LungCancer.tsv')
    aggregate_vector = aggregate.composite_embedding(entity_type, endpoint, model_list)
    # Make semEP-node executable
    subprocess.run(['chmod', '+x', 'semEP-node'], check=True)

    # Verify that semEP-node is executable
    # print(subprocess.run(['ls', '-l', 'semEP-node'], check=True, capture_output=True, text=True).stdout)

    for m in model_list:
        """Load KGE model"""
        df_donor = aggregate_vector.loc[aggregate_vector.Model==m]
        df_donor.drop(columns=['Model'], inplace=True)
        complex_numb = False
        if m == 'RotatE':
            complex_numb = True
            # Create a new DataFrame to store real parts
            real_df = pd.DataFrame()
            # Iterate over columns of the original DataFrame
            for col in df_donor.columns:
                # Skip 'Entity' column
                if col == 'Entity':
                    real_df[col] = df_donor[col]
                    continue
                # Extract real parts and store in the new DataFrame
                real_df[col] = df_donor[col].apply(extract_real_part)
            df_donor = real_df.copy()
        """Load ClinicalRecord responses file"""
        target = get_target(kg, target_predicate, df_donor)
        """Labeling donors in the DataFrame"""
        df_donor = pd.merge(df_donor, target, on="Entity")
        file_address = 'clusteringMeasures/' + m + '/'
        path_plot = '../Plots/'+kg_name+'/' + m + '/'
        for th in threshold:
            cls_address = file_address + 'SemEP_' + str(th) + '/'
            # cls_address_metis = file_address + 'METIS_' + str(th) + '/'
    
            update_cluster_folder(cls_address)
            """Create similarity matrix of Donors"""
            sim_matrix, percentile, list_sim = Utility.matrix_similarity(df_donor.drop(columns=['Relapse']),
                                                                         th, complex_numb)  # cosine_sim, euclidean_distance
            Utility.SemEP_structure(file_address + 'matrix_ClinicalRecord.tsv', sim_matrix, sep=' ')
            sim_matrix.to_csv(file_address + 'matrix_sim.txt', index=False, float_format='%.5f', mode='w+', header=False)
            Utility.create_entitie(sim_matrix.columns.to_list(), file_address + 'ClinicalRecord.txt')
            """Execute SemEP"""
            num_cls = call_semEP(percentile, cls_address, file_address)
            """METIS"""
            # update_cluster_folder(cls_address_metis)
            # if num_cls > 1:
            #     nodes = METIS_Undirected_MAX_based_similarity_graph(sim_matrix, cls_address_metis)
            #     call_metis(num_cls, nodes, cls_address_metis)
            """Labeling donors in the matrix"""
            sim_matrix = sim_matrix.merge(target, left_index=True, right_on='Entity', suffixes=('_df1', '_df2'))
    
            cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)],
                                          index=['No_Progression', 'Progression', 'Relapse',
           'UnKnown'])
            entries = os.listdir(cls_address + 'clusters/')
            for file in entries:
                sim_matrix.loc[
                    sim_matrix.Entity.isin(Utility.load_cluster(file, cls_address + 'clusters/')), 'cluster'] = int(
                    file[:-4].split('-')[1])
                df_donor.loc[
                    df_donor.Entity.isin(Utility.load_cluster(file, cls_address + 'clusters/')), 'cluster'] = int(
                    file[:-4].split('-')[1])
            """Compute statistics for each cluster"""
            cluster_statistics(sim_matrix.drop(['Entity'], axis=1), cls_statistics, num_cls, cls_address)
    
            if not os.path.exists(path_plot):
                os.makedirs(path_plot)
            if len(entries) < 9:
                new_df = Utility.plot_semEP(len(entries), sim_matrix.drop(['Entity'], axis=1), path_plot, 'PCA_th_' + str(th) + 'matrix.pdf',
                                            scale=False, show=False)
                new_df[['Relapse', 'cluster']].to_csv(path_plot + 'th_' + str(th) + '_summary.csv')
                Utility.plot_semEP(len(entries), df_donor.drop(columns=['Entity']), path_plot, 'PCA_th_' + str(th) + '.pdf',
                                   scale=False, show=False)
            df_donor.drop(columns=['cluster'], inplace=True)
    
            """Execute Kmeans"""
            # sim_matrix['ClinicalRecord'] = sim_matrix.index
            sim_matrix.drop(columns=['cluster'], inplace=True)
            kmeans_address = file_address + 'Kmeans_' + str(th) + '/'
            if not os.path.exists(kmeans_address):
                os.makedirs(kmeans_address)
            # num_cls = Utility.elbow_KMeans(sim_matrix.iloc[:, :-2], 1, 15, kmeans_address)  # df_donor
            # if num_cls is None:
            #     num_cls = 15
            new_df, cls_report = Utility.plot_cluster(num_cls, sim_matrix, kmeans_address, scale=False, show=False)  # df_donor
            new_df.to_csv(kmeans_address + 'cluster.csv', index=None)
            update_cluster_folder(kmeans_address)
            """Save Kmeans-Clusters"""
            for cls in range(num_cls):
                new_df.loc[new_df.cluster == cls][['Entity']].to_csv(
                    kmeans_address + 'clusters/' + 'cluster-' + str(cls) + '.txt', index=None, header=None)
            """Compute statistics for each cluster"""
            cls_statistics = pd.DataFrame(columns=['cluster-' + str(x) for x in range(num_cls)],
                                          index=['No_Progression', 'Progression', 'Relapse',
           'UnKnown'])
            cluster_statistics(new_df, cls_statistics, num_cls, kmeans_address)
            print('Community Detection by SemEP and Kmeans, considering ' +m+ ' model and threshold '+ str(th)+ ' done.')

        """Visualize Donors"""
        Utility.plot_treatment(df_donor, path_plot)
        """Density of Donor Similarity"""
        Utility.density_plot(list_sim, path_plot)
        return aggregate_vector