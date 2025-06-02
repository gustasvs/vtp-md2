## 1 Install
```
pip install -r requirements.txt
```  
```
python -m spacy download en_core_web_sm
```

## 2 Data
### download datasets
```
python data_fetch.py
```     
### cleanup data and tokenise     
```
python data_preprocess.py
```

## 3 Models
script                 | purpose                  | output  
-----------------------|--------------------------|--------------------------  
model_naive_bayes.py   | 3-fold CV + train        | nb_ekman.pkl  
model_roberta.py       | fine-tune RoBERTa-models | roberta_model/

*(Optionally run `python main.py` to chain steps 2-3)*
### To grid search Naive Bayes model
Set variable `ANALYSE` to `True` in `model_naive_bayes.py` file
### To test different RoBERTa models
Change variable `MODEL_NAME` in `model_roberta.py` file

## 4 Inference
For Bayes model: ```python model_naive_bayes.py```

For RoBERTa model:
```python model_roberta.py``` with uncommented `#inference()` in main script

