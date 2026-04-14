# Useful videos

### 1. Fine-Tuning BERT for Text Classification (w/ Example Code)
#### Here a dude describes steps for using BERT for downstream tasks such as loading data, freezing parameters, data pre-processing, etc. https://www.youtube.com/watch?v=4QHg8Ix8WWQ

### 2. Huggingface links
#### https://huggingface.co/google-bert/bert-large-uncased - larger BERT model
#### https://huggingface.co/datasets/stanfordnlp/imdb - imdb dataset (Labels 1 and 0. 1 for Positive and 0 for Negative)

### 3. BERT is encoder only, unlike latest LLMs like GPT or Gemini, which are Decoder only.

1) First Encoder Only Transformer creates **word embeddings** to predict word sequences. **Word Embeddings converts words, bits of words and symbols collectively called **Tokens**, into numbers. **But it does not take into account word order.
2) Thus, **Positional Encoding**. It helps to keep track of word order.
3) To keep track of the **relationships** among words we use **Self-Attention**. (Encoder-only transformers use Self-Attention). 
   Self-attention works by seeing how similar each word is to all the words in the sentence, including itself.
4) All of the above helps create a new type of embedding called **Context Aware Embedding**.

### Deeper notes about BERT
1. Tokens and Initial Embeddings
BERT uses a method called WordPiece tokenization.

Tokens: It doesn't just look at whole words. It breaks rare words into "bits" (sub-words). For example, "unhappy" might become un and ##happy. This prevents the "out-of-vocabulary" problem.

The "Static" Start: At the very first layer, these tokens are converted into vectors (numbers). You're right—at this specific stage, the vector for "bank" (a river bank) and "bank" (a financial institution) are identical. They don't have context yet.

2. Positional Encoding
Because Transformers process all words in a sentence simultaneously (parallelization) rather than one by one (like older RNNs), the model is naturally "order-blind."

The Solution: BERT adds a "positional vector" to the initial embedding.

Correction: Unlike the original Transformer which used fixed mathematical formulas (sine/cosine waves), BERT actually learns these positional embeddings during its training. It’s like giving each word a seat number in a theater so the model knows who is sitting next to whom.

3. Self-Attention (The "Relationship" Engine)
This is the "brain" of the operation.

In technical terms, Self-Attention uses three vectors for every token: Query (Q), Key (K), and Value (V).

The model asks: "How much should the word 'it' (Query) focus on the word 'robot' (Key)?"

It calculates a score, and if the score is high, it pulls more information from the "Value" of that word.

The "B" in BERT: Unlike GPT (which only looks at previous words), BERT is Bidirectional. It looks at words to the left and the right simultaneously to understand the full relationship.

### Difference between Encoder only and Decoder only
1. Directionality: The Core Difference
Encoder-Only (e.g., BERT)
Encoders are Bidirectional. When BERT looks at a word in a sentence, it can see the words that come before it and the words that come after it at the same time.

The Goal: To create the most accurate mathematical representation of a specific word based on its surroundings.

Analogy: It’s like a student reading a whole paragraph multiple times to fully grasp the deep meaning before answering a multiple-choice question.

Decoder-Only (e.g., GPT-4, Llama)
Decoders are Unidirectional (or "Causal"). They are restricted: they can only see the words that came before. They are literally "masked" from seeing the future.

The Goal: To predict the very next token in a sequence.

Analogy: It’s like a person writing a story one word at a time. They know what they’ve written so far, but the "future" words haven't been created yet.

Benefits of Encoder-Only (The "Analyst")
Better for Classification: Because it sees the whole sentence at once, it is much better at picking up subtle sentiment cues. In your sentiment analysis task, BERT will likely outperform a GPT-style model of the same size because it doesn't have "blind spots" regarding the end of the sentence.

Efficiency: They are often smaller and faster to run for specific tasks like categorizing emails or spotting spam.

Benefits of Decoder-Only (The "Creator")
Zero-Shot Learning: These models are incredibly good at following instructions they’ve never seen before (e.g., "Write a poem about a toaster in the style of Shakespeare").

Open-Ended Generation: They can keep writing for pages while maintaining a consistent "train of thought." Encoders can't really "write" original content; they can only label or transform what's already there.