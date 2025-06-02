## 1 Install
pip install -r requirements.txt  
python -m spacy download en_core_web_sm

## 2 Data
python data_fetch.py          # download datasets
python data_preprocess.py     # cleanup data and tokenise

## 3 Models
script                 | purpose                  | output  
-----------------------|--------------------------|--------------------------  
model_naive_bayes.py   | 3-fold CV + train        | nb_ekman.pkl  
model_roberta.py       | fine-tune RoBERTa-models | roberta_large_emotion/

*(Run `python main.py` to chain steps 2-3.)*

## 4 Inference
python model_naive_bayes.py                 # interactive loop  
python -c "from model_roberta import inference; inference()"
```
