# This is a sample Python script.
import random
import json
import pickle
import numpy
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model

import speech_recognition as sp

rec = sp.Recognizer()

my_micro = sp.Microphone(device_index=1)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('chatbot.json').read())
info = json.loads(open('info.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(words)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

print(words)

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.model')
print('Done')

wrds = pickle.load(open('words.pkl', 'rb'))
clss = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intents_list[0]['intent']:
            res = random.choice(i['responses'])
            break
    print(res)
    if 'Da li' in res and '/' not in res and 'ili' not in res:
        if input("").lower() == 'da':
            pom = res.lower().split(" ")
            pom = pom[len(pom) - 1].split("?")[0]
            print(urls(pom))
    elif '/' in res or 'ili' in res:
        print(urls(input("")))
    elif ':' in res:
        print(urls(res.lower().split(" ")[0]))

def urls(information):
    for i in info['info']:
        if information in i['type']:
            return i['url']
    return ""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("KAKO ZELIS DA ZAPOCNES RAZGOVOR?")
    print("1.GOVOR")
    print("2.DOPISIVANJE")
    if input("") is "1":
        while True:
            rec = sp.Recognizer()

            my_micro = sp.Microphone(device_index=1)

            with my_micro as source:
                print("Reci nesto")
                audio = rec.listen(source)

                try:
                    message = rec.recognize_google(audio, None, "sr-SP", False)
                except:
                    continue
                print(message)
                ints = predict_class(message.lower())
                get_response(ints, intents)
    else:
        while True:
            print("--")
            message = input("")
            ints = predict_class(message.lower())
            get_response(ints, intents)
