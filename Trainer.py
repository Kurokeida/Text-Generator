import spacy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from pickle import dump,load
import numpy as np

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1198623

# Function to add all tokens except ones included in the if not list
def seperate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

d = read_file('< >')
tokens = seperate_punc(d)

# organize into sequences of tokens
train_len = 25+1 # 50 training words , then one target word

# Empty list of sequences
text_sequences = []

for i in range(train_len, len(tokens)):
    
    # Grab train_len# amount of characters
    seq = tokens[i-train_len:i]
    
    # Add to list of sequences
    text_sequences.append(seq)

' '.join(text_sequences[0])
' '.join(text_sequences[1])
' '.join(text_sequences[2])

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

vocabulary_size = len(tokenizer.word_counts)

#Converting sequences into numpy arrays
sequences = np.array(sequences)

X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y,num_classes=vocabulary_size+1)

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model

model = create_model(vocabulary_size+1,seq_len)

model.fit(X, y, batch_size=128, epochs=0,verbose=1)

model.save('<Name>.h5')
dump(tokenizer,open('<Name>','wb'))
