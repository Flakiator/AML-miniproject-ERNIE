# AML-miniproject-ERNIE

## GloVe + LSTM (glove.ipynb)
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


## Data Analyis (eda.ipynb)
### Motivation
The `eda.ipynb` file is what we used to answer some of our questions regarding the data. Mainly:
1. Word frequency
2. Token frequency
3. Distribution of sentence length in characters
4. Distribution of sentence length in words

These enabled us to pinpoint what data preprocessing we should do in our pipeline, notably:
- Finding that some reviews exceed BERT's max input size
- The data included HTML tags

### Running it
It can be run in colab or locally. Running it will, at the bottom, result in the 4 aforementioned graphs.
