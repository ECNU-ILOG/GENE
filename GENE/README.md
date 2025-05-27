# GENE: a generic causal discovery method for both strongly and weakly identifiable problems under ANM.
## How to Run
```python
conda create -n GENE python==3.8
pip install -r requirements.txt
python run.py --node_num 10 --density 2 --sem mim --sample_size 3000
