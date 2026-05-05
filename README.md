# AML-miniproject-ERNIE
This project is for the Advanced Machine Learning Course of Spring 2026 (KSAMLDS2KU).
The students involved are:
- Marcus Andreas Aandahl (maraa)
- Niklas Zeeberg Hessner Christensen (nizc)
- Anton Yakovenko (anya)

The project focuses on the analysis of different ML models as a means to sentiment analysis of movie reviews on the IMDB dataset.
It more notably investigates the use of BERT and GloVe variants, and how each can capture context in natural languages, within the scope of movie reviews.

## Data
The data is the IMDB movie review dataset (in english).
It contains a test set and a validation set, each with 25.000 reviews.
Each set is evenly split 50/50 between positive and negative reviews.
It contains a large plethora of tokens, both with HTML tags, made up words, and spelling mistakes.

## Method chosen
We initially wanted to use the BERT model, as it provided great bi-directional context encoding, which we felt was relevent to capture the sentiment
of the reviews.
After implementing a basic BERT model, we started playing around with different parameters and techniques, such as weights freezing and decay,
to see how it'd affect our model.

Having some success with BERT, we proceeded to implement a baseline approach that we could compare the BERT implementation with.
We decided on using GloVe vector embeddings along with a few hidden layers. We expected that it'd provide some good naive predictions, but
fails for more contextually convoluted reviews.

Being mostly interested in the capturing of context, this is what we then decided to play with is order to compare models.

## Architecture
### BERT
TODO
- architecture
- training mechanisms
- brief justification (for non-standard things)

### GloVe
TODO
- architecture
- training mechanisms
- brief justification (for non-standard things)

### GloVe + LSTM
We also implemented a GloVe model where we try to add some notion of bi-directional context encoding.
It starts with an embedding layer, which matches word indexes in the 
This is done via a bi-directional LSTM layer, which then concatenates the result of going each direction, before pooling it all.
The idea here is:
1. 

## Results
ANTON
- key experiments and results
- accuracy tables
- error graphs (maybe from wandb?)

Key experiments & results: present and explain results, e.g. in simple accuracy tables over error graphs up to visualisations of representations and/or edge cases – keep it crisp

## Discussion
ANTON
TODO

Discussion: summarise the most important results and lessons learned (what is good, what can be improved)

## How to run the models and other resources
### GloVe + LSTM (glove.ipynb)
The GloVe + LSTM `glove.ipynb` can be run in colab, where we recommend using T4 runtime. It scores a test accuracy of around 88%.

The first cell should look as follows during first run, as to download the GloVe vectors (uncomment 2nd and 3rd lines):
```py
# Downloads (run once)
# !wget -q http://nlp.stanford.edu/data/glove.6B.zip
# !unzip -q glove.6B.zip -d glove

# V1: Test accuracy: 0.8774 | Test loss: 0.2897
# V2: Test accuracy: 0.8828 | Test loss: 0.2795
# V3: Test accuracy: 0.8830 | Test loss: 0.2795
```

Thereafter, the code should be fairly well annotated. Once all cells are run once, the first many cells can be skipped, as the model is saved.

If the model is saved, it can be loaded by running the second to last cell.

Some prediction examples are also added, along with a symbol dictating how well we found the prediction to be.

### Data Analyis (eda.ipynb)
#### Motivation
The `eda.ipynb` file is what we used to answer some of our questions regarding the data. Mainly:
1. Word frequency
2. Token frequency
3. Distribution of sentence length in characters
4. Distribution of sentence length in words

These enabled us to pinpoint what data preprocessing we should do in our pipeline, notably:
- Finding that some reviews exceed BERT's max input size
- The data included HTML tags

#### Running it
It can be run in colab or locally. Running it will, at the bottom, result in the 4 aforementioned graphs.
