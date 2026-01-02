Source code: bap.py\
Model weights: /models\
Results (Section A): /results\
Metrics (Section B): /metrics

# CSE 559 Final Project Repository

This repo contains the source code, model weights, and loss histories for each run, and mean and std of AUC/Precision/Recall/F1 scores. Only the seed 42 model weights were saved due to memory.

To run:
Put your .pkl file containing embeddings inside the same directory as bap.py (the project root). The script assumes this relative path. All output files will be written to that directory.

```
python bap.py
```

If you use embeddings other than BLOSUM, the "dat = pd.read_pickle(file_path)" line can require >30GB of CPU RAM. This is more than the limits of most computers and Google Colab's T4.

Installation details are shown below, using Python 3.12. 

```bash
git clone https://github.com/edwin-huang/CSE559TCREpitopeProject
cd CSE559TCREpitopeProject
python3.12 -m venv bap
source bap/bin/activate
python -m pip install --upgrade pip
pip install pandas==2.2.2 scikit-learn==1.6.1 tqdm==4.67.1 torch==2.9.0 cuml-cu12==25.10.0
```

If the pip install line does not work, you can use
```
pip install pandas scikit-learn tqdm torch cuml-cu12
```
Compatibility is not guaranteed.

If you are using a CUDA version that is not 12, you can use the cuml version specific to the CUDA version. Compatibility is not guaranteed.