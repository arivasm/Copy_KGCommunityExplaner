[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Symbolic Learning and KG Embeddings with Link Prediction and Community Detection
The objective of this workshop is to highlight the impact of symbolic learning over various downstream KG embeddings tasks such as Link Prediction (LP) and Community Detection. For reproducibility, follow the instructions provided in the [slides](https://docs.google.com/presentation/d/14EOHzqKmI2KpseAif_GWtCi2cBEb1ksTlZMVVqFy9cQ/edit?usp=drive_link). If running locally, then follow the steps below. 


### Symbolic Learning
Follow the instructions to execute symbolic learning to generate Enriched, and Transformed KG.

```json
{
  "KG": "OriginalKG",
  "prefix": "http://example.org/lungCancer/entity/",
  "rules_file": "LungCancer-rules-short.csv",
  "rdf_file": "LungCancer.nt",
  "constraints_folder": "Constraints"
}
```

```python
python symbolic_predictions.py
```
### Knowledge Graph Embeddings and Link Prediction

```python
python SymbolicLearning_KGE/KGEmbedding/kge.py --dataset_path "SymbolicLearning_KGE/KG/OriginalKG/LungCancer.tsv" --output_dir "SymbolicLearning_KGE/KGEmbedding/OriginalKG" --results_path "SymbolicLearning_KGE/KGEmbedding/OriginalKG/" --models TransH
 ```
### Community Detection







## License
This work is licensed under the MIT license.

### Authors
Tutorial has been implemented in joint work by Yashrajsinh Chudasama, Disha Purohit, and Ariam Rivas.


