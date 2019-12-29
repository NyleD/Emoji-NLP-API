# EmojifyTextBackend
Sentiment Analysis NLP API to add emojies to everyday texts

#### How it Works
The api takes your text messages and assign's the most relevant emojies to the end of each sentence. We didn't just assign emojies to differnt words, we understood that this a NLP task. So with the use of LSTM, our algorithm can assign emojies based on the sequence of words and the meaning of the whole sentence.

#### Front End
https://github.com/NyleD/EmojiText

#### How to Use It?
You can place a POST request at https://emojifyapi.herokuapp.com/emojify/api/v1.0


#### Technologies:

- Flask
- Keras/Tensorflow
- Django (Website)
- LSTM
