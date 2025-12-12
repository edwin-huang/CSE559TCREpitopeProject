This repo contains the source code and model weights, loss histories for each run, and AUC/Precision/Recall/F1 scores for each run. Only 1 run out each embedding/split grouop was saved for model weights, as each model takes up 32MB of space.

To run:
Put your .pkl file containing embeddings inside the same directory as bap.py (the project root). The script assumes this relative path. All output files will be written to that directory.

```
python bap.py
```

If you use embeddings other than BLOSUM, the "dat = pd.read_pickle(file_path)" line can require >20GB of CPU RAM. This is more than the limits of most computers and Google Colab's T4.

Installation details are shown below, using Python 3.12. 

Linux (intended)
```bash
git clone https://github.com/edwin-huang/CSE559TCREpitopeProject
cd CSE559TCREpitopeProject
python3.12 -m venv bap
source bap/bin/activate
python -m pip install --upgrade pip
pip install pandas==2.2.2 scikit-learn==1.6.1 tqdm==4.67.1 torch==2.9.0
```

If the pip install line does not work, you can use
```
pip install pandas scikit-learn tqdm torch
```
Compatibility is not guaranteed.

For Windows, replace these lines
```bash
python3.12 -m venv bap
source bap/bin/activate
```
with:

Command Prompt
```cmd
py -3.12 -m venv bap
bap\Scripts\activate
```
Powershell (may require changing execution policies)
```powershell
py -3.12 -m venv bap
bap\Scripts\Activate.ps1
```