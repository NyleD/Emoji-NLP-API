# Emoji NLP API
Sentiment Analysis NLP API to add emojies to everyday texts

#### How it Works
The api takes your text messages and assign's the most relevant emojies to the end of each sentence. It doesn't just assign emojies to differnt words. This is a NLP task, so with the use of LSTM, the algorithm can assign emojies based on the sequence of words and the meaning of the whole sentence.

#### Chat Demo
To see how it's used in a live chat please check out
https://github.com/NyleD/EmojiChat-Backend

#### How to Use It?
You can place a POST request at https://emojifyapi.herokuapp.com/emojify/api/v1.0. Unfortuntely, the word vectors and training set only consists of real words, so the API will not work with slang words. However, I am currently improving the training set, to accompany slang words and texts from different parts of the world, so everyone can use it! 

#### What does the api return? 
A json object, where each value repersents a sentence. Each key is the ith number for the ith sentence starting at 0 in string format. For Example: "I love this site. I am so happy." would return the following object. {'0': 'I love this site 😄', '1': 'I am so happy 😄'}

#### Technologies:
- Flask
- Keras/Tensorflow
- LSTM
