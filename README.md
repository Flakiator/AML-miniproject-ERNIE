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
We used pretrained GloVe (6B, 300d) embeddings as a baseline to compare against BERT. For each review, we tokenized the text by lowercasing and whitespace splitting, looked up each token's 300-dimensional GloVe vector, and averaged them into a single fixed-size sentence representation via mean pooling. This vector was then passed into a small feedforward classifier (300 → 256 → 2) trained with AdamW.
#### The key limitation of this approach is that GloVe embeddings are static — every word always maps to the same vector regardless of context. This means the model cannot distinguish sentiment that depends on context, and word order is lost entirely through mean pooling
#### GloVe + small MLP classifier 
In our project we used pre-trained word vectors trained with GloVe (6B tokens, 400K vocab, 300d).

Our training process can be separated into two distinct stages:
1. Stage 1 is creation of **embeddings** of IMDB dataset. 
   1. We tokenize raw text.
   2. Mean-pool using GloVe vectors that produce 300d vector for a review.
2. Stage 2 is our MLP Classifier.
   1. We pass 300d vector through first Linear layer and reduce dimensions to 256.
   2. Then we use RELU activation function.
   3. Apply dropout regularization.
   4. Finally, a final Linear layer to get last 2 dimensions.
   5. We get two raw logits and apply CrossEnthropy loss that internally applies softmax to turn these numbers into probabilities.


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
#### After training our models, we decided to check their accuracy and competency on a defined set of prompts. Starting at easy-to-classify review to more convoluted.
##### **BERT** implementation correctly predicted most reviews with varying confidence, but it failed on really long reviews because of limited context length (512 tokens).
##### **Simple GloVe** did have a few correct predictions, but it completely failed on more ambiguous reviews, here the lack of contextual awareness shines the most. For example words that describe positivity/negativity just add up for the final prediction. If review had 2 negative words and 1 positive, review would become negative. So overall performance was pretty bad.
##### **Glove + LSTM** performed much better than Simple GloVe because of added context encodings and additional hidden layers to extract additional features. This version of GloVe implementation showed much better results in comparison to Simple GloVe and very close to BERT implementation. Even besting BERT in long review classification.

## Discussion
We learned that understanding the data and dataset is really important. This step is of utmost importance as it can hinder any further development. By thoroughly choosing and refining your data you will achieve much more accuracy of trained model and significantly reduce time trying to optimize it.
Furthermore, we investigated evolution of NLP approaches and the motivation behind constant improvement by learning about shortcomings and negatives of different model architectures.
Finally, we learned different methodologies and tactics for improving models and that it is possible to achieve high scores and performance using already pre-trained models.

## How to run the models and other resources
### BERT (bert.ipynb)
The colab file (BERT.ipynb) is set up in a way that you can skip certain blocks depending, like loading and old model if you are running for the first time. 
It's easy to follow instructions and run the training of a model.
You can also skip printing embeddings, tokenization and data-collection.
There are options to load old model, freeze weights and further customization as selecting custom hyperparameters and logging options.

### GloVe + LSTM (glove.ipynb)
The GloVe + LSTM `glove.ipynb` can be run in colab, where we recommend using T4 runtime. It scores a test accuracy of around 88%.

### Simple Glove (Simple_GloVe.ipynb / Simple_GloVe.py)
.ipynb version can be run in colab like Glove + LSTM version.
.py version can be run locally if your machine is faster than colab.
We used the glove.6B.zip (300d) from: https://nlp.stanford.edu/projects/glove/

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
