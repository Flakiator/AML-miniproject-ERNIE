# SETUP WEEK 09APR-15APR
The goal is to have an great overall understanding of the project as to be able to make decision and move on with the practicaly part.

## New Plan
- [x] Understand BERT (weekend)
- [x] Play around with BERT (tuesday)
- [x] Look at data (tuesday)
- [x] Import the data (tuesday)
- [x] Define classes for the output (thursday after having all findings)

## Understanding BERT
BERT is a open source NLP model, and uses Bidirectional Encoder Representations from Transformers.
It represents text as a sequence of vectors.
It uses self-supervised learning.
It uses the encoder-only transformer architecture.

It is trained using:
- masked token prediction
- next sequence prediction

As such, it learns contextual latent features of tokens in their context.

There are multiple versions of it:
- TINY with 4M parameters
- BASE with 110M parameters
- LARGE with 340M parameters

It is trained on the Toronto BookCorpus (800M words) and English Wikipedia (2.500M words) datasets.

### Encoder-only Transformer Architecture
The model consists of these modules:
- Embedding =>
    - converts token sequence into real-valued vectors, also reducing dimensionality
    - these vectors have n dimensions, where each dimension encodes some feature of the word
- Positional Encoder =>
    - vector addition is performed to encode the word positions into the current word embeddings/vectors
    - this is done since word position does affect context
- Self-Attention =>
    - encodes the context into the words
    - for example, if the word "it" is present, the noun it refers to will enventually be encoded in it
      as such, the word "it" will have the features/embeddings that the word it refers to has

This results in context aware embeddings, taking position and surrounding words into account for each word.

### Sources
- [BERT - huggingface.co](https://huggingface.co/blog/bert-101#2-how-does-bert-work)
- [BERT - Github](https://github.com/google-research/bert)
- [BERT - Twitter sentiment analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)

