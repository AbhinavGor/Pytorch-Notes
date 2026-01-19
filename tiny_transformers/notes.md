# 2026/01/07

## Questions / initial understanding
1. What do the dimensions [B, T, D] signify? 
   1. I know that B is the batch size. But I am confused about what T and D signify.
2. In MHA, when we divide into h number of attention heads, why do we divide then on dimention D? I think fully answering question 1 will answer this question.
   1. I still do not see the reason why we want to divide (maybe since we are using multiple attention heads, we are dividing the latent dimensionality amongst these "h" heads.)
3. What is the point of MHA?
   1. I understand the point of Self-Attention. With Self-Attention we introduce the possiblilty of the model including dependencies with far-off elements like in RNNs, but we ensure the computations are parallelizable.
   2. I did not understand why we would want to have multiple attention heads with smaller D in [B, T, D] for each head.

## Updated Understanding (After reading / chatgpt / answers)
1. What do the dimensions [B, T, D] signify? 
   1. "T" signifies the number of tokens or the length of each input sequence. For example, if the input is "I love cats", and we tokenize this input sequence to 3 tokens, then the value of T will be = 3.
   2. D is the latent dimension. This is the number of dimensions we want to *encode* our input token to.
2. In MHA, when we divide into h number of attention heads, why do we divide then on dimention D? I think fully answering question 1 will answer this question. and What is the point of MHA?
   1. We *have* to divide the input the third dimension in [B, T, D] i.e along "D". This is because we want all the attention heads to see the same sequence, hence, T cannot be divided. We also do not want to divide B because we want all the attention to be trained on all the batches.
   2. We want to divide into multiple attention heads because each attention heads looks at the input sequence in a different manner, one might want to put more emphasis on local connections, one might want to add more weight to long range connections.