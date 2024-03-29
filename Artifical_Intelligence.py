import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence
import spacy
import tkinter as tk
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

df = pd.read_csv("Dataset.csv") #Load csv file into data frame

train_df, test_df = train_test_split(df, test_size=0.2) #Split the data frames, with the test_size being =0.2

#Function to preprocess text (removing numbers and special characters, tokenizing and lowering words)
def preprocess_text(text):
    text = text.lower() #convert to lower case
    text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])#Remove punctuation
    tokens = word_tokenize(text)#tokenize
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return ' '.join(tokens)  # Join tokens into a single string

#Apply pre processing
train_df['text'] = train_df['text'].apply(preprocess_text)

test_df['text'] = test_df['text'].apply(preprocess_text)

train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce').round().astype('Int64')

#Drop nulled labels
train_df = train_df.dropna(subset=['label'])

#Convert to int (originally floats)
test_df['label'] = pd.to_numeric(test_df['label'], errors='coerce').round().astype('Int64')

test_df = test_df.dropna(subset=['label'])

#Convert to CSV Files, without having an index column
train_df[['text', 'label']].to_csv("trainDataset.csv", index=False)
test_df[['text', 'label']].to_csv("testDataset.csv", index=False)

#While loop to run the program until the user exits
while(True):
    op = input("1- Naive Bayes\n2- GRU\n3- LSTM\n")
    op = int(op)
    if(op==1):
        train_data = pd.read_csv('trainDataset.csv')

        test_data = pd.read_csv('testDataset.csv')

        X_train, y_train = train_data['text'], train_data['label']
        X_test, y_test = test_data['text'], test_data['label']

        vectorizer = CountVectorizer() #Initialize the vectorizer, uses bag of words representation 
        X_train_vectorized = vectorizer.fit_transform(X_train) #creates the vocabulary and fits it in an array
        #that represents frequency of words
        X_test_vectorized = vectorizer.transform(X_test) #same for test data

        naive_bayes_classifier = MultinomialNB() #probability technique, calculates the probability of
        #a word given a class, depending on its frequency 
        naive_bayes_classifier.fit(X_train_vectorized, y_train)

        predictions = naive_bayes_classifier.predict(X_test_vectorized)

        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.2f}")

        print("Classification Report:\n", classification_report(y_test, predictions))

        def on_predict_button_click():
            user_input = entry.get()
            if user_input.strip() == "":
                messagebox.showwarning("Input Error", "Please enter a sentence.")
                return
            user_input = vectorizer.transform([user_input])
            prediction = naive_bayes_classifier.predict(user_input)
            result_label.config(text=f"Sentiment Prediction: {prediction[0]}")

        # Create the main application window
        app = tk.Tk()
        app.title("Sentiment Analysis GUI")
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.2f}")

        label2 = tk.Label(app,text = f"Classification Report:\n{classification_report(y_test, predictions)}\n Accuracy: {accuracy:.2f}")
        # Create and pack widgets
        label2.pack(pady=10)
        label = tk.Label(app, text="Enter a sentence:")
        label.pack(pady=10)

        entry = tk.Entry(app, width=50)
        entry.pack(pady=10)

        predict_button = tk.Button(app, text="Predict Sentiment", command=on_predict_button_click)
        predict_button.pack(pady=10)

        result_label = tk.Label(app, text="")
        result_label.pack(pady=10)

        # Run the application
        app.mainloop()
    elif(op==2):
            start_time = time.time()

            spacy_en = spacy.load("en_core_web_sm")  #Load spacy_en for tokenization 

            #Define the fields, what each "sample" consists of (text and a label)
            #Include lengths in text since its sequential data, dont do it for label
            TEXT = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)], include_lengths=True)
            LABEL = Field(sequential=False, use_vocab=False)
            fields = [('text', TEXT), ('label', LABEL)] 
            #Split the data into train_data and test_data, skip the header of the csv file
            train_data, test_data = TabularDataset.splits( 
                path='',  #Path to the csv files
                train='trainDataset.csv',  
                test='testDataset.csv',   
                format='csv',
                fields=fields,
                skip_header=True 
            )

            #Build the vocabulary for the data, only for words that appear at least once
            TEXT.build_vocab(train_data, min_freq=1)

            #initialize device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device for the process
            print(device)

            #Split into train_iterator and test_iterator, and put 8 in the same batch and sort by length
            #Splitting them into batches will cause padding
            train_iterator, test_iterator = BucketIterator.splits(
                (train_data, test_data),
                batch_size=8, #Split data into batches, takes 8 data samples and splits them
                sort_key=lambda x: len(x.text), #Sort texts by length
                sort_within_batch=True,
                #Shuffle the data
                shuffle=True, 
                device=device #Set the device
            )
            #Initialize the model
            class GRUModel(nn.Module):
                def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
                    super().__init__()
                    #word embedding, depending on vocab size
                    self.embeddingLayer = nn.Embedding(vocab_size, embedding_dim) 
                    #the GRU layer
                    self.gruLayer = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
                    #Linear layer
                    self.linearLayer = nn.Linear(hidden_dim, output_dim)
                    #Dropout layer to reduce overfitting
                    self.dropout = nn.Dropout(dropout)

                def forward(self, text, text_lengths):
                    #embed the data
                    embedded = self.embeddingLayer(text)
                    #Remove the padding
                    packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu())
                    _, hidden = self.gruLayer(packed_embedded)
                    #Drop the last hidden state
                    hidden = self.dropout(hidden[-1, :, :])
                    return self.linearLayer(hidden)

            #Initialize the values
            vocab_size = len(TEXT.vocab)
            embedding_dim = 150
            hidden_dim = 256 
            output_dim = 1
            n_layers = 3
            dropout = 0.3
            #Call the model
            model = GRUModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
            errorCalculator = nn.BCEWithLogitsLoss() #Binary cross entropy => Calculates error
            #Adam optimizer, used to update the model parameters during training mode with learning rate =0.001
            optimizer = optim.Adam(model.parameters())

            num_epochs = 10

            for epoch in range(num_epochs):
                model.train() #put the model in training mode
                total_loss = 0

                for batch in train_iterator:
                    text, text_lengths = batch.text
                    labels = batch.label.float() #convert labels into floats

                    optimizer.zero_grad() #Zero the error gradients before using them to update weights
                    predictions = model(text, text_lengths).squeeze(1) #Remove any extra singleton caused by padding

                    
                    #flatten the dimensions
                    predictions = predictions.view(-1)
                    labels = labels.view(-1)

                    #Calculate the loss after making the predictions, change the weights in backward propagation 
                    loss = errorCalculator(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                average_loss = total_loss / len(train_iterator)
                print(f'Training Epoch: {epoch + 1}, Loss: {average_loss}')

            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = len(test_data)
                test_loss = 0
                total_TP=0
                total_TN=0
                total_FP = 0
                total_FN= 0

                for batch in test_iterator:
                    text, text_lengths = batch.text
                    labels = batch.label.float()

                    #Predict the output
                    predictions = model(text, text_lengths).squeeze(dim=1) #remove extra dimensions caused by padding

                    #flatten predictions and labels to be able to calculate the loss
                    predictions = predictions.view(-1)
                    labels = labels.view(-1)

                    #calculate the loss
                    loss = errorCalculator(predictions, labels)
                    test_loss += loss.item()

                    #Check prediction
                    binary_predictions = (torch.sigmoid(predictions) >= 0.5).float()
                    correct = (binary_predictions == labels).sum().item()
                    total_correct += correct

                    #calculate the sum and convert it into integer
                    tp = ((binary_predictions == 1) & (labels == 1)).sum().item()
                    tn = ((binary_predictions == 0) & (labels == 0)).sum().item()
                    fp = ((binary_predictions == 1) & (labels == 0)).sum().item()
                    fn = ((binary_predictions == 0) & (labels == 1)).sum().item()

                    total_TP += tp
                    total_TN += tn
                    total_FP += fp
                    total_FN += fn

                print(total_correct)
                print(total_samples)
                print(f'TP:{total_TP}\n')
                print(f'TN:{total_TN}\n')
                print(f'FP:{total_FP}\n')
                print(f'FN:{total_FN}\n')
                average_test_loss = test_loss / len(test_iterator)
                accuracy = total_correct / total_samples
                Precision= total_TP/(total_TP+total_FP)
                Recall= total_TP/(total_TP+total_FN)
                F1_Score= 2*Precision*Recall/(Precision + Recall)
                print(f'Test Results after all epochs - Loss: {average_test_loss}, Accuracy: {accuracy * 100:.2f}%')
                print(f'Precision: {Precision}\n')
                print(f'Recall: {Recall}\n')
                print(f'F1_Score: {F1_Score}\n')


            print("Training and testing complete!")
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Elapsed time: {elapsed_time/60} minutes")
            def predict_sentiment(model, tokenizer, text):
                model.eval()
                with torch.no_grad():
                    tokenized_text = tokenizer(text)
                    indexed_text = [TEXT.vocab.stoi[token] for token in tokenized_text]
                    length = len(indexed_text)
                    tensor_text = torch.LongTensor(indexed_text).view(length, 1).to(device)
                    text_lengths = torch.tensor([length]).to(device)

                    prediction = torch.sigmoid(model(tensor_text, text_lengths)).item()
                    return prediction

            # Function to handle button click
            def on_predict_button_click():
                user_input = entry.get()
                if user_input.strip() == "":
                    messagebox.showwarning("Input Error", "Please enter a sentence.")
                    return

                prediction = predict_sentiment(model, lambda x: [tok.text for tok in spacy_en.tokenizer(x)], user_input)
                result_label.config(text=f"Sentiment Prediction: {prediction:.4f}")

            # Create the main application window
            app = tk.Tk()
            app.title("Sentiment Analysis GUI")

            label2 = tk.Label(app,text = f'Test Results after all epochs - Loss: {average_test_loss}, Accuracy: {accuracy * 100:.2f}%')
            # Create and pack widgets
            label2.pack(pady=10)
            label = tk.Label(app, text="Enter a sentence:")
            label.pack(pady=10)

            entry = tk.Entry(app, width=50)
            entry.pack(pady=10)

            predict_button = tk.Button(app, text="Predict Sentiment", command=on_predict_button_click)
            predict_button.pack(pady=10)

            result_label = tk.Label(app, text="")
            result_label.pack(pady=10)

            # Run the application
            app.mainloop()
    elif(op==3):
            start_time = time.time()
            # Load the pre-processing file and execute it

            spacy_en = spacy.load("en_core_web_sm")  # Load spacy_en for tokenization 

            # Define the fields, what each "sample" consists of (text and a label)
            # Include lengths in text since it's sequential data, don't do it for the label
            TEXT = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)], include_lengths=True)
            LABEL = Field(sequential=False, use_vocab=False)
            fields = [('text', TEXT), ('label', LABEL)] 

            # Split the data into train_data and test_data, skip the header of the CSV file
            train_data, test_data = TabularDataset.splits( 
                path='',  # Path to the CSV files
                train='trainDataset.csv',  
                test='testDataset.csv',   
                format='csv',
                fields=fields,
                skip_header=True 
            )

            # Build the vocabulary for the data, only for words that appear at least once
            TEXT.build_vocab(train_data, min_freq=1)

            # Initialize device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device for the process
            print(device)

            # Split into train_iterator and test_iterator
            train_iterator, test_iterator = BucketIterator.splits(
                (train_data, test_data),
                batch_size=8,  # Split data into batches, takes 8 data samples and splits them
                sort_key=lambda x: len(x.text),  # Sort texts by length
                sort_within_batch=True,
                # Shuffle the data
                shuffle=True, 
                device=device  # Set the device
            )

            # Initialize the model with LSTM
            class LSTMModel(nn.Module):
                def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
                    super().__init__()
                    # Word embedding, depending on vocab size
                    self.embedding = nn.Embedding(vocab_size, embedding_dim) 
                    # The LSTM layer
                    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
                    # Linear layer
                    self.fc = nn.Linear(hidden_dim, output_dim)
                    # Dropout layer to reduce overfitting
                    self.dropout = nn.Dropout(dropout)

                def forward(self, text, text_lengths):
                    # Embed the data
                    embedded = self.embedding(text)
                    # Make sure that they have the same embed length (set the rest to 0), packed based on their lengths
                    packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu())
                    _, (hidden, _) = self.lstm(packed_embedded)
                    # Drop the last hidden state
                    hidden = self.dropout(hidden[-1, :, :])
                    return self.fc(hidden)

            # Initialize the values
            vocab_size = len(TEXT.vocab)
            embedding_dim = 150
            hidden_dim = 256 
            output_dim = 1
            n_layers = 3
            dropout = 0.3

            # Call the model
            model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

            errorCalculator = nn.BCEWithLogitsLoss()  # Binary cross-entropy
            # Adam optimizer, used to update the model parameters during training mode with learning rate = 0.001
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            num_epochs = 10

            for epoch in range(num_epochs):
                model.train() #put the model in training mode
                total_loss = 0

                for batch in train_iterator:
                    text, text_lengths = batch.text
                    labels = batch.label.float() #convert labels into floats

                    optimizer.zero_grad() #Zero the error gradients before using them to update weights
                    predictions = model(text, text_lengths).squeeze(1) #Remove any extra singleton caused by padding

                    
                    #flatten the dimensions
                    predictions = predictions.view(-1)
                    labels = labels.view(-1)

                    #Calculate the loss after making the predictions, change the weights in backward propagation 
                    loss = errorCalculator(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                average_loss = total_loss / len(train_iterator)
                print(f'Training Epoch: {epoch + 1}, Loss: {average_loss}')

            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = len(test_data)
                test_loss = 0
                total_TP=0
                total_TN=0
                total_FP = 0
                total_FN= 0

                for batch in test_iterator:
                    text, text_lengths = batch.text
                    labels = batch.label.float()

                    #Predict the output
                    predictions = model(text, text_lengths).squeeze(dim=1) #remove extra dimensions caused by padding

                    #flatten predictions and labels to be able to calculate the loss
                    predictions = predictions.view(-1)
                    labels = labels.view(-1)

                    #calculate the loss
                    loss = errorCalculator(predictions, labels)
                    test_loss += loss.item()

                    #Check prediction
                    binary_predictions = (torch.sigmoid(predictions) >= 0.5).float()
                    correct = (binary_predictions == labels).sum().item()
                    total_correct += correct

                    #calculate the sum and convert it into integer
                    tp = ((binary_predictions == 1) & (labels == 1)).sum().item()
                    tn = ((binary_predictions == 0) & (labels == 0)).sum().item()
                    fp = ((binary_predictions == 1) & (labels == 0)).sum().item()
                    fn = ((binary_predictions == 0) & (labels == 1)).sum().item()

                    total_TP += tp
                    total_TN += tn
                    total_FP += fp
                    total_FN += fn

                print(total_correct)
                print(total_samples)
                print(f'TP:{total_TP}\n')
                print(f'TN:{total_TN}\n')
                print(f'FP:{total_FP}\n')
                print(f'FN:{total_FN}\n')
                average_test_loss = test_loss / len(test_iterator)
                accuracy = total_correct / total_samples
                Precision= total_TP/(total_TP+total_FP)
                Recall= total_TP/(total_TP+total_FN)
                F1_Score= 2*Precision*Recall/(Precision + Recall)
                print(f'Test Results after all epochs - Loss: {average_test_loss}, Accuracy: {accuracy * 100:.2f}%')
                print(f'Precision: {Precision}\n')
                print(f'Recall: {Recall}\n')
                print(f'F1_Score: {F1_Score}\n')


            print("Training and testing complete!")
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Elapsed time: {elapsed_time/60} minutes")
            def predict_sentiment(model, tokenizer, text):
                model.eval()
                with torch.no_grad():
                    tokenized_text = tokenizer(text)
                    indexed_text = [TEXT.vocab.stoi[token] for token in tokenized_text]
                    length = len(indexed_text)
                    tensor_text = torch.LongTensor(indexed_text).view(length, 1).to(device)
                    text_lengths = torch.tensor([length]).to(device)

                    prediction = torch.sigmoid(model(tensor_text, text_lengths)).item()
                    return prediction

            # Function to handle button click
            def on_predict_button_click():
                user_input = entry.get()
                if user_input.strip() == "":
                    messagebox.showwarning("Input Error", "Please enter a sentence.")
                    return

                prediction = predict_sentiment(model, lambda x: [tok.text for tok in spacy_en.tokenizer(x)], user_input)
                result_label.config(text=f"Sentiment Prediction: {prediction:.4f}")

            # Create the main application window
            app = tk.Tk()
            app.title("Sentiment Analysis GUI")

            label2 = tk.Label(app,text = f'Test Results after all epochs - Loss: {average_test_loss}, Accuracy: {accuracy * 100:.2f}%')
            # Create and pack widgets
            label2.pack(pady=10)
            label = tk.Label(app, text="Enter a sentence:")
            label.pack(pady=10)

            entry = tk.Entry(app, width=50)
            entry.pack(pady=10)

            predict_button = tk.Button(app, text="Predict Sentiment", command=on_predict_button_click)
            predict_button.pack(pady=10)

            result_label = tk.Label(app, text="")
            result_label.pack(pady=10)

            # Run the application
            app.mainloop()
    else:
            break
