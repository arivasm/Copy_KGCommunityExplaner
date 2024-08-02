import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import fitz  # PyMuPDF
from PIL import Image
import io
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the directory containing ComputeCommunities.py
relative_path = '../PatternDetection'
# Construct the absolute path to the directory
path_to_script = os.path.abspath(os.path.join(current_dir, relative_path))
# Add this directory to the system path
sys.path.append(path_to_script)
import ComputeCommunities as SemCD

def target_cluster(kg_name, model, target_predicate, cls_algorithm, th):
    kg = SemCD.get_kg('../KG/' + kg_name + '/LungCancer.tsv')
    """Load KGE model"""
    path_model = '../KGEmbedding/' + kg_name + '/'
    df_donor = pd.read_csv(path_model + model + '/embedding_donors.csv')
    """Load ClinicalRecord responses file"""
    target = SemCD.get_target(kg, target_predicate, df_donor)
    path = '../PatternDetection/clusteringMeasures/' + model + '/' + cls_algorithm + '_' + str(th) + '/clusters/'

    list_donor = []
    entries = os.listdir(path)
    for file in entries:
        cls = pd.read_csv(path + file, delimiter="\t", header=None)
        cls.columns = ['ClinicalRecord']
        target.loc[target.ClinicalRecord.isin(cls.ClinicalRecord), 'Community'] = 'Community ' + file[:-4].split('-')[1]
        list_donor = list_donor + list(cls.ClinicalRecord)

    target = target.loc[target.ClinicalRecord.isin(list_donor)]
    replacement_mapping_dict = {'No_Progression': 'No relapse',
                                'Relapse': 'Relapse',
                                'Progression': 'Relapse'}
    target['Relapse'].replace(replacement_mapping_dict, inplace=True)
    return target


def catplot(df_reset, model):
    g = sns.catplot(df_reset, kind="bar",
        x="Community", y="count_values", hue='Relapse',
                    height=6, aspect=1.2, palette=['#264653', '#2A9D8F', '#E9C46A'])
    legend = g._legend  # Access the legend object
    # legend.set_title("Legend Title")  # Set the legend title
    # Set the legend's fontsize and other properties
    legend.get_title().set_fontsize(16)  # Set the title font size
    legend.get_texts()[0].set_fontsize(14)  # Set the label font size for the first item
    legend.get_texts()[1].set_fontsize(14)  # Set the label font size for the second item
    # legend.get_texts()[2].set_fontsize(14)  # Set the label font size for the second item
    # Change the legend position
    legend.set_bbox_to_anchor((0.6, 0.85))  # Adjust the position as needed

    g.set_axis_labels("", "Normalized Clinical Records", fontsize=16)
    plt.title('Distribution of Relapse by Community', fontsize=16)
    # ax.set_ylabel("Parameter values",fontsize=16)
    plt.tick_params(labelsize=16)
    plt.ylim(0, .9)
    # plt.savefig('Plots/Kmeans_norm_v2.pdf', bbox_inches='tight', format='pdf', transparent=True)
    # plt.savefig('Plots/METIS_norm_v2.pdf', bbox_inches='tight', format='pdf', transparent=True)
    plt.savefig(model+'.pdf', bbox_inches='tight', format='pdf', transparent=True)


def PCA_projection(kg_name, model, threshold):
    # Path to the PDF file
    pdf_path1 = '../Plots/' + kg_name + '/' + model + '/PCA.pdf'
    pdf_path2 = '../Plots/' + kg_name + '/' + model + '/PCA_th_' + str(threshold) + '.pdf'

    # Open the PDF files
    fig1 = fitz.open(pdf_path1)
    fig2 = fitz.open(pdf_path2)

    # Select the page number (0-based index)
    page_number1 = 0
    page_number2 = 0

    # Load the pages
    page1 = fig1.load_page(page_number1)
    page2 = fig2.load_page(page_number2)

    # Render the pages to images
    pix1 = page1.get_pixmap()
    pix2 = page2.get_pixmap()

    # Convert the images to PIL Images
    img1 = Image.open(io.BytesIO(pix1.tobytes()))
    img2 = Image.open(io.BytesIO(pix2.tobytes()))

    # Plot the images side by side using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Display the first image
    axes[0].imshow(img1)
    axes[0].axis('off')  # Hide the axis
    # axes[0].set_title('Figure 1')

    # Display the second image
    axes[1].imshow(img2)
    axes[1].axis('off')  # Hide the axis
    # axes[1].set_title('Figure 2')
    plt.show()