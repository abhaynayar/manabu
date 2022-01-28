# Sequence Models
## RNN

```
TBD
```

## Word Embeddings
### Word Representation
- 1-hot representation.
- Featurized representation: word embeddings.

### Analogies

```
e.man - e.woman = e.king - e.w
w: argmax sim(e.w, e.king - e.man + e.woman)
```

Cosine similarity:

```
sim(u,v) = np.dot(u.T,v) / |u|.|v|
```

### Learning word embeddings

- Multiply word embeddings matrix (all words and their features) with one
word's 1-hot vector (position of word in vocabulary).
- Use it for multiple  words at once and throw them into a neural network.
- No need to use all words that came before. Use last 4 words for context.
- Or use last four and next four words for context.

### Word2Vec

Skip-grams:

- Instead of last 4 words, randomly pick a context word.
- Randomly choose another word within a window of 5 words as the target.
- **Expensive softmax objective step.**

```
Softmax: p(t|c) = np.exp(theta.b.T, e.c) / np.sum(np.exp(theta.j.T, e.c))
Loss:    L = -np.sum(np.dot(y,log(y')))
```

Negative sampling:

- Take a sentence. Take a context (c) and target (t) word after it. Label
  (y) it true.
- Take the same context (c) word and sample a random target (t) word from
  the vocabulary. Label (y) it false. Number of negative samples is k.
- k=5-20 for smaller datasets, k=2-5 for larger datasets.

Model:

```
P(y=1|c,t) = sigmoid(np.dot(theta.t.T, e.c)
```

### GloVe Word Vectors

Xij = number of times j appears in the context of i

Model:

```
for i in range(10000):
    for j in range(10000):
        sum += f(Xij).(np.dot(theta.i.T, e.j) + bi + bj' - log(Xij))**2

f(Xij) = 0 if Xij=0
minimize sum
```

Then:

```
e.w(final) = (e.w + theta.w)/2
```

## Sequence Models & Attention Mechanism

### Sequence to sequence models

- We have an encoder and a decoder.
- Can be used for machine translation and image captioning.

### Machine translational as building a conditional language model

- In a language model, given one word, we find the probability of the most
  likely next word.
- In machine translation, we can take the sentence from the source language
  and pass it through a decoder, and then pass it through another network
  called the encoder which is like the language model.
- So machine translation is almost like language models, except for instead
  of starting from zeros, we start with the output of the encoder network.

## Beam Search

- Has parameter B (beam width), which decides how man likely words to
  choose at a time.
- If B=3, at every step you instantiate three copies of the network.
- Refinements:
  - Length normalization:
    - Use sum of logs of probabilities.
    - Divide the sum by number of words. (Ty)
    - Or also raise Ty to the power of alpha. (heuristic alpha=0.7)
  - How to choose Beam Width (B): large, better but slower. Try till
    diminishing returns.
- Error analysis: where's the error, beam search or the RNN?
  - Human = y\*, Algorithm = y^
  - Case 1: P(y\* | x) > P(y^ | x)  =>  Beam search is at fault.
  - Case 2: P(y\* | x) <= P(y^ | x)  =>  RNN is at fault.


## Attention Model
### Intuition

The problem of long sequences: in RNNs we encode a sentence and using those
encodings, we generate a decoded sentence. This works fine for shorter
sentences, but not for more than 30 words. How much information can you
expect that encoding to hold?

So instead, we use the attention model to translate things as and when they
occur. So for very large texts, we keep translating as we keep reading just
like what a human would do.

For every word, we want to calculating how much attention should we be
paying to every other word.

### Implementation

Attention values are non-negative.
Sum of all attentions will be 1.
Context will be sum of attention\*activation.

### How to compute attention weights

Take previous state s\<t-1\> and activation \<t'\> as inputs to a small
neural network, and train it to get e\<t,t'\>. Then we get the attention
values (denoted by alpha) through the energies (e) and activations (a)
inserted into a formula.

## Speech Recognition

We first create a spectogram. At each time step, how loud is it at all
frequencies?

- Academic datasets might use 300 to 3000 hours of audio.
- Best commercial datasets might be 10,000 to 100,000 hours long.

CTC cost method:
- Outputs a bunch of characters.
- Collapse repeated characters not separated by "blank".

## Trigger Word Detection

As soon as the audio finishes saying the trigger word, output labels 1's in
the neural network.



## Transformer Network
### Intuition

- Attention + CNN
  - Self-attention
  - Multi-head attention

### Self-attention

- A(q,K,V) = attention-based vector representation of a word.
- Calculate A for each word.

RNN attention:

```py
TBD
```

Transformer attention:

```py
A(q,K,V) = softmax(np.dot(Q,K.T)/np.sqrt(dk)) * V
```

- q=query, K=key, V=value.
- To calculate A\<3\>:
  - Compute the inner product between the q3 and the keys of all words.
  - Then using those values, put them into the softmax given above.
  - Then multiply output of that softmax with values of all words.
  - Sum up all those outputs.
  - You can write A\<3\> = A(q\<3\>, K, V)


### Multi-head attention

- Like a for loop over self-attention.
- Each iteration will ask a different question.
- However, we are still looping over the same word.
- All the heads can be computed in parallel.


## Transformer Network

- Multi-head attention (Q,K,V) inside an encoder block.
- Produces a matrix which can be passed into a feed forward neural network.
- Repeat this encoder block N times. (N=6)
- Feed output of this into a decoder block.
- Decoder block takes target language sentence up till now and puts it into
  a multi-head attention block. This block outputs Q.
- We put the Q from before and K and V from encoder output into another
  multi-head attention block. We then put this in a feed forward neural
  network which outputs a prediction for the next word.
- We take this next word and put this back into the input for the decoder
  block as the translated sentence up till now.


## Transformer details

- Need positional encodings.
- Some weird sin and cos waves into a vector.
- Positional encoding is added to the input word vectors X.
- Masked multi-head attention: used during training, blocks out last part
  of the sentence.


