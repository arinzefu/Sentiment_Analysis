
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("movie.csv")

df.head()

df.shap
# Initialize label_counts dictionary
label_counts = {'0': 0, '1': 0}

# Iterate over rows of DataFrame and count labels
for index, row in df.iterrows():
    label = row['label']
    label_counts[str(label)] += 1

# Print label counts
print(label_counts)

# Preprocess data
df['text'] = df['text'].apply(lambda x: x.lower())  # Lowercase text
df['text'] = df['text'].str.replace('[^\w\s]', '', regex=False)  # Remove punctuation

from sklearn.model_selection import train_test_split

# Split into train and test sets
train_size = 0.7
test_size = 0.15
val_size = 0.15

train_df, test_df = train_test_split(df, test_size=test_size+val_size, random_state=42)
test_df, val_df = train_test_split(test_df, test_size=val_size/(test_size+val_size), random_state=42)

print(train_df)

import matplotlib.pyplot as plt

# Calculate the length of each sequence
sequence_lengths = [len(seq.split()) for seq in df['text']]

# Plot the distribution of sequence lengths
plt.hist(sequence_lengths, bins=50)
plt.show()

from gensim.models import Word2Vec

# Train the Word2Vec model
corpus = [doc.split() for doc in train_df['text']]
Word2Vecmodel = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, workers=4)
# Tokenize text data
tokenizer = Tokenizer(num_words=50000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['text'])

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
val_sequences = tokenizer.texts_to_sequences(val_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Pad sequences
train_padded = pad_sequences(train_sequences, maxlen=256, truncating='post', padding='post')
val_padded = pad_sequences(val_sequences, maxlen=256, truncating='post', padding='post')
test_padded = pad_sequences(test_sequences, maxlen=256, truncating='post', padding='post')

# Define the vocabulary size and embedding matrix
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, 100))  # Initialize embedding matrix
for word, i in word_index.items():
    if word in Word2Vecmodel.wv.key_to_index:
        embedding_matrix[i] = Word2Vecmodel.wv[word]


#define hypermaraters
embedding_dim = 200
max_length = 256
num_classes = 2

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()


from keras.utils import to_categorical

# Convert labels to one-hot encoded matrix
train_labels = to_categorical(train_df['label'])
val_labels = to_categorical(val_df['label'])

# Train the model
history = model.fit(train_padded, train_labels, validation_data=(val_padded, val_labels), epochs=5, batch_size=16)

# Evaluate the model on the test set
test_labels = to_categorical(test_df['label'])
test_loss, test_acc = model.evaluate(test_padded, test_labels)
print('Test accuracy:', test_acc)
model.save('sentiment_model.h5')

# predict sentiment for a given text
text = "This movie was so poorly written and directed I fell asleep 30 minutes through the movie."
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
probability = model.predict(padded_sequence)[0][0]

# determine sentiment based on probability
if probability > 0.5:
    print("Negative")
else:
    print("Positive")

import gradio as gr

# # Creating a Gradio App

# Define the labels
labels = {0: 'Negative', 1: 'Positive'}
# Define the function to predict the sentiment
def predict_sentiment(text):
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=120, padding='post', truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)
    prediction_label = labels[np.argmax(prediction)]

    # Return the prediction
    if prediction[0][0] > 0.5:
        return "Negative"
    else:
        return "Positive"

# Create the Gradio interface
input_text = gr.inputs.Textbox(lines=5, label='Enter Text')
output_text = gr.outputs.Textbox(label='Sentiment')

gr.Interface(fn=predict_sentiment, inputs=input_text, outputs=output_text, title='Sentiment Analysis For Movie Review',
             description='Enter A Movie Review and get the Sentiment Analysis result',
             theme='Full Page', analytics_enabled=True).launch(share=True);

