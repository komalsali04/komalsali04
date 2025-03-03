# **RNN for Text Translation**

Recurrent Neural Network is a sequential neural network that processes information step by step to capture the previous inputs. It has the ability to capture memory to use for future predictions. It can be used for tasks that require sequential processing such as speech recognition, natural language processing, and time series forecasting. 
The neural network was not able to capture the memory to predict the output at future time step due to which it lacked, and that's when the RNN came into picture.

## **Architecture of RNN**

![image](https://github.com/user-attachments/assets/65d8e2c0-10ef-43c0-87be-4a7105ca8c09)

## **Working of RNN**

Example: If we have to predict the future stock price, we need to have history of stock prices and input at current time step. RNN uses a feedback loop to store the past information as well as the input at current time step to predict the stock price at future time step.
RNN has hidden states which uses previous hidden state as memory and current input to produce the output.


**Let us train an RNN for Text Translation.** 

There is an encoder block and a decoder block in RNN and each block has it's own vocabulary. We are translating English to Hindi, so encoder will have english vocabulary and decoder will contain hindi vocabulary. These vocabularies contains the tokens(words) and each token will have its own token ID.

**Let's follow these steps to convert an English sentence to Hindi:**

### **Encoder Block:**

1. Convert the english tokens into embeddings(numerical representation of tokens).
2. Initialize the hidden state of the encoder, this vector is initialised with zeros.
3. Initialize the W, U, and B vector. These are trainable parameters of RNN and can be adjusted.
4. Define a mathematical relation between these vector to compute the next hidden state.
   ht = tanh(W * ht-1 + U * xt + B)
   Here, ht = Hidden state, ht - 1 = previous hidden state, and xt = current input
   The inputs are embeddings of the tokens, eg: Token = "I", embedding = [0.1]
5. Compute all hidden states of encoder, once the final hidden state is computed, we transfer the content of final hidden state to the decoder block.

### **Decoder Block:**

1. Convert all hindi tokens to embeddings(numberical representation of tokens).
2. The initial hidden state (1st hidden state of decoder block) is equal to the final hidden state of encoder block.
3. There are two additional tokens used in decoder block, <GO> and <EOS> tokens, where the <GO> token indicates that the hidden state is ready for translation and it waits until all the content from final hidden state of encoder is tranferred to the first hidden state of decoder, and the <EOS> token tells the model to stop decoding.
4. The inputs in decoder block are the translated token of previous hidden state since there are no explicit inputs in decoder block.
5. Now, we have the first hidden state of decoder block, initialise the vector Wdec, Udec, and Bdec, which are trainable patterns of RNN and can be adjusted.
6. Define the mathematical relation between hidden states to compute each hidden state.
   ht = tanh(Wdec * ht - 1 + Udec * xt + Bdec)
   The working of decoder block is similar to the encoder.
7. Once the hidden state is computed, we compute the logits matrix which is a vector of probability and is used to determine what the next token will be.
   The values inside logits matrix corresponds to the tokens inside output vocabulary(Hindi vocabulary).
   To compute the logits matrix, we need to consider two more matrices, Wout and Bout. The dimension of Wout is equal to (output vocabulary size * hidden state size), where output vocabulary size is the dimension of output vocabulary and hidden state size is the dimension of hidden state. The dimension of Bout is equal to the dimension of hidden state.
   logits matrix = Wout * hidden state + Bout
   After we obtain the logits matrix, apply softmax activation function over logits matrix to obtain the final vector of probability. The softmax function is used to ensure that the logits matrix has all positive values, as the values are replaced by exponents, and ensure that it boosts the confidence of the model to make predictions. The final logits matrix has probabilities that adds to 1 across the dimension.
   The highest value in the logits matrix corresponds to the next token.
8. This is how all the hidden states are computed and the tokens are translated. Once the <EOS> is generated as next token, it indicates the end of decoding.

## **Drawback of RNN:**

1. **Loss of Long-Term context in final hidden state:** As all the previous hidden states are forwarded to the next hidden states of RNN, the final hidden state is bombarded with all the previous inputs which makes it impossible for the model to convert large data into single vector. Due to this, the model is likely to lose some amount of data.
   
2. **Vanishing Gradient:** As we compute the gradient of model in every epoch, it gets smaller and smaller and it becomes so tiny that it stops updating the parameters of the model effectively due to which, the model forget the past tokens and end up remembering the latest tokens in the sequence. Therefore, RNN cannot capture long range dependencies. Hence, it can't be trained on long sequences.


                                                                          
   
   


