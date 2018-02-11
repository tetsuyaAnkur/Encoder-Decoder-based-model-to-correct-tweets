# Encoder-Decoder-based-model-to-correct-tweets
# Preprocess
I have used the csv file "consolidated.csv" as the training dataset. It contains 17000 rows, each containing an input tweet and a corrected tweet. (this file is uploaded into the repository)

First I extracted the tweet column. I treated a tweet as a sentence and tokenized it. This gave me a sequence of words for
each tweet. I used genism Word2Vec to do this.

Then I created a corpus as per the requirements of genism Word2Vec and trained the Word2Vec model.

After this I created input feature vectors for 10K rows using the word2vec model, that I created and trained earlier.

After this I read the column labeled as "corrected". For each corrected tweet I did word tokenization. For each tweet that I tokenized I got a sequence of tokens. So,now I created a one hot vector, whose length is the size of vocabulary where the elements are all 0’s except at the index that corresponds to the word which is the expected token.

Now, for 10K rows I got 10K sequences constituting the inputs and 10K target sequences, where each target sequence represented the tokens that were predicted.

As the length of each sequence was a variable, I had to pad the sequences as needed.

# RNN Model
I have used keras to build the encoder decoder model without attention.
