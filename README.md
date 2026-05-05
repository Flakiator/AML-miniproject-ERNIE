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
We used google-bert/bert-base-uncased from HuggingFace, a 12-layer bidirectional transformer with 12 attention heads and a hidden size of 768, pretrained on masked language modelling and next sentence prediction. 

We loaded the pretrained weights via BertModel rather than BertForSequenceClassification to expose the architecture explicitly and allow flexibility in swapping the output head for different tasks. The CLS token's final hidden state is passed through a dropout layer and a linear classification head to produce sentiment logits.

For training we used the HuggingFace Trainer with AdamW as the optimizer. Weight decay was applied for regularization. Metrics were logged per epoch to Weights & Biases, and the best checkpoint was selected based on validation F1 score.

### GloVe
TODO (Anton)
- architecture
- training mechanisms
- brief justification (for non-standard things)

### GloVe + LSTM
We also implemented a GloVe model where we try to add some notion of bi-directional context encoding.
It starts with an embedding layer, which matches word indexes in the input sequence with a its GloVe embedding equivalent.
Thereafter, we have bi-directional LSTM layer, used to add some context encoding for each word both ways (each both ways are concatinated).
The result of the bi-directional LSTM layer for each word is then pooled for the whole sentence in order to aggregate the context of all the words combined.
The sentence context is then fed into a hidden layer (with ReLU activation) with the hope of encoding
any hidden feature not captured by the previous layers.
We then have a dropout layer as to prevent overfitting, before then having a sigmoid activation function in our last layer to end up with a 1/0 value
representing positive/negative respectively.

The overall idea here is:
1. Use GloVe embeddings from words
2. Encode sentence context for each word once per direction and concatinate the results
3. Pool word context to get sentence context
4. Hidden layer to capture any hidden feature missed by the bi-directional LSTM layer
5. Dropout layer to prevent overfitting
6. Output layer with sigmoid activation function

#### Training mechanisms
For training mechanisms, we first decided to freeze the GloVe embedding layer, since we felt like the dataset was not big enough to
represent the wider data population, and thus it might turn innacurrate.

We also used a dropout layer as to prevent overfitting. This works by occcasionally removing some of the encodings from the previous layers, which might otherwise get too much importance.

Furthermore, we used learning rate decay in order to improve fine-tuning and not overshoot optimization.
We also added early stopping, such that when multiple epochs without improvements is experience, the model reverts to the best weights before
it started to do bad.

The optimizer used was Adaptive Moment Estimation with a base LR of 0.001 (and used LR decay as mentioned above).
The loss was calculated via binary cross-entropy, since we were working with binary output classes.

## Results
TODO (Anton)
- key experiments and results
- accuracy tables
- error graphs (maybe from wandb?)

Key experiments & results: present and explain results, e.g. in simple accuracy tables over error graphs up to visualisations of representations and/or edge cases – keep it crisp

## Discussion
TODO (Anton)

Discussion: summarise the most important results and lessons learned (what is good, what can be improved)

## How to run the models and other resources
### GloVe + LSTM (glove.ipynb)
The GloVe + LSTM `glove.ipynb` can be run in colab, where we recommend using T4 runtime. It scores a test accuracy of around 88%.

The first cell should look as follows during first run, as to download the GloVe vectors (uncomment 2nd and 3rd lines):
```py
# Downloads (run once)
!wget -q http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip -d glove

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
