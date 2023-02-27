# GIST NLP 2022 Spring Assignment


### Assignment 1 - Sequential Labeling Input Transform
*kaggle link : https://www.kaggle.com/competitions/nlp-lab1-problem1/overview*

Sequential labeling is a form of most of the problems to obtain special information in natural languages we use, such as Korean and English, and is a representative problem to solve symbols with abstract meaning in machine learning. Data consisting of arbitrary symbols and labels are provided to verify the ability to construct a model for the format of the problem, rather than solving it with analysis of special domain-specific knowledge or qualities. It verifies the ability to implement a method of configuring a model to receive sequential input of symbols.

The purpose of this problem is to verify the ability to implement the part of connecting variable symbol sequences to the model after the vectorization process from basic file input to use them in neural network models. To simplify the verification environment, the model implements a 3-layer feedforward network consisting of the following techniques: An input is a long vector with vectors representing symbols attached (concatenate).


### Assignment 2 - Sentence Classification
*kaggle link : https://www.kaggle.com/competitions/nlp-lab2*

To predict the label of input sentence, first convert each character in the word, which is tokenized and lemmatized, to dense vector. After character embedding, convert each word in the sentence by using it. Next, convert sentences to vector by using 1D CNN and finally, train classifier and get the prediction about the input sentences

### Assignment 3 - Classification & POS Tagging
*kaggle link : https://www.kaggle.com/competitions/nlp-lab3-problem1*\n
*kaggle link : https://www.kaggle.com/competitions/nlp-lab3-problem2*

To predict the input sentence label, use given glove word embedding dictionary file. Trough embedding layer, convert each time step token to vector and input to 3 layer RNN. In the last, through the classifier layer, predict the target label on the last time step.

To predict the next time step label, use given FastText embedding dictionary file. Trough embedding layer, convert each time step token to vector and input to 3 layer bi-directional RNN for left to right pass. Next flipped input and do same process for right to left pass. Concatenate last layer hidden states and pass through classifier then predict the target label.

### Assignment 4 - Neural Machine Translation
*kaggle link : https://www.kaggle.com/competitions/nlp-lab4*

To translate the input sequence in source language, first preprocess the input for mini-batch. After preprocessing, implement Unidirectional LSTM-based Sequence-to-Sequnce (called Seq2Seq). In this step, you have to implement packing, encoder-decoder, autoregressive vs. non-autoregressive and teacher forcing. Next, Implement Attention mechanism on decoder, and train Autoregressive Seq2Seq with Attention model on NMT.


### Assignment 5 - Neural Machine Translation
*kaggle link : https://www.kaggle.com/competitions/nlp-lab-5*

To translate the input sequence in source language, first preprocess the input for mini-batch. After preprocessing, implement Transformer. In this step, you have to implement. 1) scale dot-product attention, 2) multi-heads attention, 3)Pad & Sub-Sequence Masks, 4) Feed Forward sub Layer, and 5) Auto-regressive Decoder.
