import pandas as pd
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
#from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from decimal import Decimal
from math import *
from sklearn.preprocessing import StandardScaler

from sklearn.tree import _tree, DecisionTreeClassifier
from IPython.display import display, HTML
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity


def density_plot(list_sim, path_plot):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    standard_similarity = pd.DataFrame()
    standard_similarity.insert(0, 'similarity', list_sim)
    # fig, ax = plt.subplots()
    sns.kdeplot(data=standard_similarity, x="similarity", fill=True, cut=0,
                bw_adjust=.1)  # ,hue='fold', legend=False,  bw_method=0.01  .25
    # move_legend(ax, "upper center")
    # plt.ylim(0,18)
    plt.xlabel('Similarity')
    plt.ylabel('Density')
    plt.savefig(path_plot + 'SimilarityDensity.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def load_cluster(name, address):
    cls = pd.read_csv(address + name, delimiter="\t", header=None)
    cls.columns = ['Entity']
    # print('cls.Entity:', list(cls.Entity))
    return list(cls.Entity)


def cosine_sim(x, y):
    return abs(1 - scipy.spatial.distance.cosine(x, y))


def euclidean_distance(x, y):
    return 1 / (1 + sqrt(sum(pow(a - b, 2) for a, b in zip(x, y))))


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def nth_root(value, n_root):
    root_value = 1 / float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def minkowski_distance(x, y, p_value):
    return nth_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)


def normalize_matrix(sim_matrix):
    max_val = max(sim_matrix.max())
    sim_matrix = 1 - sim_matrix.div(max_val)
    return sim_matrix


def matrix_similarity(embedding, th, complex_numb):
    df = embedding.set_index('Entity')
    # Extract the vectors from the DataFrame
    vectors = df.values
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    # Scale the values to the range [0, 1]
    similarity_matrix = (similarity_matrix + 1) / 2
    # To create a new DataFrame with the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
    # Convert DataFrame to NumPy array
    arr = similarity_df.values
    # Extract the lower triangular part of the array, excluding the diagonal
    lower_triangular = np.tril(arr, k=-1)
    # Extract the lower diagonal values (excluding zeros)
    lower_diag_values = lower_triangular[lower_triangular != 0]
    threshold = np.percentile(lower_diag_values, th)
    # print("percentil", threshold)
    return similarity_df, threshold, lower_diag_values


# === Save cosine similarity matrix with the structure SemEP need
def SemEP_structure(name, sim_matrix, sep):
    f = open(name, mode="w+")
    f.write(str(sim_matrix.shape[0]) + "\n")
    f.close()
    sim_matrix.to_csv(name, mode='a', sep=sep, index=False, header=False, float_format='%.5f')


def create_entitie(list_n, ENTITIES_FILE):
    # entities = "\n".join(str(x) for x in list_n)
    n_ent = str(len(list_n))
    pd.DataFrame([n_ent]+list_n).to_csv(ENTITIES_FILE, index=None, header=None)


    # entity = open(ENTITIES_FILE, mode="w+")
    # entity.write(n_ent + "\n" + entities)
    # entity.close()


def elbow_KMeans(matrix, k_min, k_max, n):
    plt.close()
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(k_min, k_max), random_state=0)
    visualizer.fit(matrix)
    num_cls = visualizer.elbow_value_
    # visualizer.show(outpath=n + "elbow.pdf", bbox_inches='tight')
    return num_cls


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def cluster_statistics(df, cls_statistics, num_cls, cls_address):
    for c in range(num_cls):
        try:
            cured = df.loc[df.cluster == c][['response']].value_counts().cured
        except AttributeError:
            cured = 0
        try:
            non_cured = df.loc[df.cluster == c][['response']].value_counts().non_cured
        except AttributeError:
            non_cured = 0
        cls_statistics.at['cured', 'cluster-' + str(c)] = int(cured)  # / 14
        cls_statistics.at['non_cured', 'cluster-' + str(c)] = int(non_cured)  # / 73
    cls_statistics.to_csv(cls_address + 'cls_statistics.csv')


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report(data: pd.DataFrame, clusters):
    # === Hyperparameter Grid Search with Cross Validation ===
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 20, num=19)]
    criterion = ['entropy', 'gini']
    min_samples_leaf = [int(x) for x in np.linspace(6, 40, num=35)]
    # Create the parameter grid based on the results of random search
    param_grid = {
        'max_depth': max_depth,
        'criterion': criterion,
        'min_samples_leaf': min_samples_leaf
    }
    # Create a based model
    clf = DecisionTreeClassifier(random_state=0)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, n_jobs=-1)
    # Fit the grid search to the data
    grid_search.fit(data, clusters)

    # Create Model
    tree = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'],
                                  min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                  criterion=grid_search.best_params_['criterion'])
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    return report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']]


def plot_cluster(num_cls, df, n, scale=False, show=False):
    new_df = df.copy()
    X = new_df.iloc[:, :-2]
    # To scale the data
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    kmeans = KMeans(n_clusters=num_cls, random_state=0, n_init='auto')
    y_cluster = kmeans.fit_predict(X)
    new_df['cluster'] = y_cluster
    # cls_report = cluster_report(new_df.iloc[:, :-3], y_cluster)
    cls_report = pd.DataFrame()
    # define and map colors
    col = list(mcolors.cnames.values())
    # col = list(mcolors.BASE_COLORS.values())
    col = col[:num_cls]
    index = list(range(num_cls))
    color_dictionary = dict(zip(index, col))
    # print(new_df, color_dictionary)
    new_df['c'] = new_df.cluster.map(color_dictionary)

    new_df['label'] = 'o'
    new_df.loc[new_df.Relapse == 'No_Progression', 'label'] = '*'
    new_df.loc[new_df.Relapse == 'Progression', 'label'] = '.'
    new_df.loc[new_df.Relapse == 'Relapse', 'label'] = '<'
    if num_cls<15:
        #####PLOT#####
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(1, figsize=(8, 8))
        # plot data
        pca = PCA(n_components=2).fit(X)
        dim_reduction = pca.transform(X)
        #     plt.scatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, marker=new_df.label, alpha=0.6, s=50)
        scatter = mscatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, s=50, m=new_df.label)

        # create a list of legend elemntes
        ## markers / records
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Community {}'.format(i + 1),
                                  markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(col)]
        # plot legend
        plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
        # title and labels
        plt.title('Communities of ClinicalRecords', loc='left', fontsize=22)
        plt.savefig(fname=n + "KMeans.pdf", format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    return new_df, cls_report


def plot_semEP(num_cls, df, path_plot, name, scale=False, show=False):
    new_df = df.copy()
    X = new_df.iloc[:, :-2]
    # To scale the data
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    # define and map colors
    # col = list(mcolors.cnames.values())
    col = list(mcolors.BASE_COLORS.values())
    col = col[:num_cls]
    index = list(range(num_cls))
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.cluster.map(color_dictionary)

    new_df['label'] = 'o'
    new_df.loc[new_df.Relapse == 'No_Progression', 'label'] = '*'
    new_df.loc[new_df.Relapse == 'Progression', 'label'] = '.'
    new_df.loc[new_df.Relapse == 'Relapse', 'label'] = '<'
    #####PLOT#####
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    pca = PCA(n_components=2).fit(X)
    dim_reduction = pca.transform(X)
    #     plt.scatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, marker=new_df.label, alpha=0.6, s=50)
    scatter = mscatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, s=50, m=new_df.label)

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Community {}'.format(i + 1),
                              markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(col)]
    # plot legend
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    # title and labels
    plt.title('Communities of ClinicalRecords', loc='left', fontsize=22)
    plt.savefig(fname=path_plot + name, format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return new_df


def plot_treatment(df, name, show=False):
    new_df = df.copy()
    X = new_df.iloc[:, :-2]
    col = [mcolors.CSS4_COLORS['brown'], mcolors.CSS4_COLORS['lightcoral'], mcolors.CSS4_COLORS['olive'], mcolors.CSS4_COLORS['lime']]
    index = ['No_Progression', 'Progression', 'Relapse', 'UnKnown']
    #     # index = ['test', 'train']
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.Relapse.map(color_dictionary)

    #####PLOT#####
    fig, ax = plt.subplots(1, figsize=(8, 8))
    pca = PCA(n_components=2).fit(X)
    dim_reduction = pca.transform(X)

    #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2500, random_state=42)
    #     dim_reduction = tsne.fit_transform(X)

    # plt.scatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, marker=new_df.label, s=50)  # alpha=0.6,

    scatter = mscatter(dim_reduction[:, 0], dim_reduction[:, 1], c=new_df.c, s=50)  # , m=new_df.label

    #     vocab = list(new_df.donor.values)
    #     for i, word in enumerate(vocab):
    #         plt.annotate(word, xy=(dim_reduction[i, 0], dim_reduction[i, 1]))

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=mcolor, markersize=10) for key, mcolor in color_dictionary.items()]
    #     legend_elements = []
    #     new_df = new_df[['response', 'label', 'c']]
    #     new_df.drop_duplicates(inplace=True)
    #     # print(new_df)
    #     for index, row in new_df.iterrows():
    #         if row.label == 'o':
    #             if row.response == 'effective':
    #                 legend_elements.append(Line2D([0], [0], marker='o', color='w', label='test_'+row.response,
    #                                               markerfacecolor=row.c, markersize=8))
    #             else:
    #                 legend_elements.append(Line2D([0], [0], marker='o', color='w', label='test_' + row.response,
    #                                               markerfacecolor=row.c, markersize=8))
    #         else:
    #             if row.response == 'effective':
    #                 legend_elements.append(Line2D([0], [0], marker='*', color='w', label='train_'+row.response,
    #                                               markerfacecolor=row.c, markersize=8))
    #             else:
    #                 legend_elements.append(Line2D([0], [0], marker='*', color='w', label='train_' + row.response,
    #                                               markerfacecolor=row.c, markersize=8))

    # plot legend
    plt.legend(handles=legend_elements, loc='lower left', fontsize=12)
    # title and labels
    plt.title('ClinicalRecords in P4-LUCAT', loc='left', fontsize=22)
    # plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".png", format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(fname=name + 'PCA.pdf', format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
