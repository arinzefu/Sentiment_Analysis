# Sentiment_Analysis
 
I will start with the setbacks; the dataset has labels 0: Negative and 1: positive but the model read it in reverse.
I believed it was a minor issue so I did not retrain it to change that, instead I implemented  if prediction[0][0] > 0.5:
        return "Negative"
    else:
        return "Positive"

Another setback was the test accuracy was 88% percent, I could have trained longer and gotten >95% but I did not
I tried a lot of texts and i got the right result but 'this movie is interesting' produced negative when it is positive.

The model's size was 220 mb was too big to be pushed to github. The public web interface link is only available from 09/05/2023 -12/05/2023.

This script offers a step-by-step guide on building a sentiment analysis model for movie reviews. 
It imports several essential libraries such as pandas, tensorflow, numpy, and gradio, which are required for building and training the model.

The data preprocessing phase involves removing all punctuation from the text data, splitting the data into training, validation, and test sets, and tokenizing the text data using Keras' Tokenizer. 
The tokenization process involves breaking the text data into individual words or tokens, which help the model understand the structure of sentences and the relationships between the words. The preprocessed data is then ready for building the model.

The model utilizes Word2Vec, a neural network model that maps each word in a text corpus to a high-dimensional vector, to embed the text data. 
This step helps the model understand the context of the words and their relationships with each other. 
Additionally, the model uses a convolutional neural network (CNN) to classify the reviews. CNNs are a type of neural network that are well-suited for image and text classification tasks, making them ideal for the structure of the sentences in the movie reviews.

The final step in the process is integrating the model with Gradio, a Python library that creates a user-friendly web interface.
With the Gradio web interface, users can input their own text and receive the predicted sentiment of the review. The interface is designed to be intuitive and easy to use, even for people without machine learning experience. 
By following the steps in this script, users can build their own sentiment analysis model for movie reviews and gain insights into people's emotions and opinions towards different movies.


About the Dataset.

It is a dataset from Kaggle and the URL is https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis