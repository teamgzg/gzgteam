***KÜTÜPHANELER***

#!pip install gradio

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
import re
import string
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import gradio as gr

VERİ SETİNİ OKU

data = pd.read_csv("teknofest_train_final.csv",sep="|")
data.head()

VERİ SETİNİ SAYISAL OLARAK İNCELE

text = data['text'].values
len(text)

data.target.value_counts()

data.is_offensive.value_counts()

data.info()

KATEGORİLERİ GÖRSELLEŞTİR

data.target.value_counts().plot.bar(x="Target",y="Number", color="purple", figsize=(4,5))

HER BİR SATIRDAKİ KELİME SAYISANA BAK

data['words'] = [len(x.split()) for x in data['text'].tolist()]
data[['text','words']]

 #Veri setinde kullanılan kelimelerin istatistiği
data['words'].describe()

#Her bir kategori için kullanılan kelime sayısının istatistikleri
data.groupby(['target'])['words'].describe()

GEREKSİZ VERİLERİ SİLMEK İÇİN YAPILANLAR

#Kelime sayısı '0' ve 1 olanlar 
min_size = 1
data[data['words'] <= min_size].count()

sıfır_birler = data[data['words'] <= min_size]

sıfır_birler[:50]

sıfır_birler[50:100]

sıfır_birler[100:150]

sıfır_birler[150:165]

sıfır_birler[12:-2]

#Veriler silindi
data.drop(sıfır_birler.index[12:-2], inplace=True)

#İndex değerleri değiştiği için resetlendi
data = data.reset_index(drop=True)

len(data)

VERİ SETİNDE EN ÇOK GEÇEN KELİMELERİ BULMAK İÇİN YAPILANLAR

word_freq = data.text.str.split(expand=True).stack().value_counts()
top_400_words = word_freq[:400]
top_400_words

top_400_words.to_csv('top_400_words.csv')

KATEGORİLERİN SAYIYIYA ÇEVRİLME İŞLEMLERİ

# Kategoriler isimlerden oluşuyor.Bunları sayıya çevirmeliyiz.
data['target']

# words'lere ihtiyacımız olmadığı için silindi
data = data.drop('words',axis=1)

data.target.value_counts()

#target sütunundaki  değerler yeni bir sütun olarak atandı
from collections import Counter

def add_column_with_unique_words(data):
    
    tokenized_data = [row.split() for row in data['target']]

    
    all_unique_words = set().union(*tokenized_data)

    
    for word in all_unique_words:
        column_name = str(word)
        data[column_name] = [row.count(word) for row in tokenized_data]

    return data

data2 = add_column_with_unique_words(data)

data2

new_data = data2.drop('target',axis=1)

new_data.head()

category = []

for col in new_data.columns:
    category.append(col)
del category[:2]
print(category)

new_data.dtypes

data3 = new_data.sample(frac=1)

data3

tr_stop_words = pd.read_csv("tr-stop-words.txt",header=None)

@tf.keras.utils.register_keras_serializable()
def standart_custom(input_text):
    lower = tf.strings.lower(input_text, encoding='utf-8') 
    no_stars = tf.strings.regex_replace(lower, "\*", " ")
    stripped_html = tf.strings.regex_replace(no_stars, "<br />", "") 
    no_numbers = tf.strings.regex_replace(stripped_html, "\w*\d\w*","") 
    no_punctuation = tf.strings.regex_replace(no_numbers,'[%s]' % re.escape(string.punctuation),'') 
    #remove stopwords
    no_stop_words =' '+no_punctuation+ ' '
    for each in tr_stop_words.values:
        no_stop_words = tf.strings.regex_replace(no_stop_words, ' '+each[0]+' ', r" ")
    no_space = tf.strings.regex_replace(no_stop_words, " +", " ")
    no_turkish_character = tf.strings.regex_replace(no_space, "ç", "c") 
    no_turkish_character = tf.strings.regex_replace(no_turkish_character, "ğ", "g")
    no_turkish_character = tf.strings.regex_replace(no_turkish_character, "ı", "i")
    no_turkish_character = tf.strings.regex_replace(no_turkish_character, "ö", "o")
    no_turkish_character = tf.strings.regex_replace(no_turkish_character, "ş", "s")
    no_turkish_character = tf.strings.regex_replace(no_turkish_character, "ü", "u")
    return no_turkish_character   

vocab_size = 40000
max_len = 8

X = data3['text']
y = data3[category].values

X.values[0]

data3.info()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42,stratify=y)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.1,stratify=y_train)

X_test.values[0]

y_test

print(len(X_train))
print(len(X_val))
print(len(X_test))

X_train.values[0]

y_train

vectorize_text = TextVectorization(
    standardize=standart_custom,
    max_tokens=vocab_size,
    output_mode = 'int',
    output_sequence_length=max_len
)

vectorize_text.adapt(X_train.values)

vectorized_text = vectorize_text(X_train.values)

vectorized_text_val = vectorize_text(X_val.values)

vectorized_text_test = vectorize_text(X_test.values)

vectorized_text_test[0]

y_test

train_size = len(X_train)

batch_size=4
AUTOTUNE=tf.data.experimental.AUTOTUNE

train = tf.data.Dataset.from_tensor_slices((vectorized_text,y_train))
train=train.shuffle(buffer_size=train_size)
train=train.batch(batch_size=batch_size,drop_remainder=True)
train=train.cache()
train = train.prefetch(AUTOTUNE)

val = tf.data.Dataset.from_tensor_slices((vectorized_text_val,y_val))
val=val.shuffle(buffer_size=train_size)
val=val.batch(batch_size=batch_size,drop_remainder=True)
val=val.cache()
val = val.prefetch(AUTOTUNE)


def create_Sequential_Model():
    model = Sequential()
    model.add(Embedding(vocab_size+1,32))
    model.add(Bidirectional(LSTM(32,activation='tanh')))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='BinaryCrossentropy',optimizer='Adam')
    return model

model = create_Sequential_Model()

history = model.fit(train, validation_data=val, epochs=7)

y_pred=model.predict(vectorized_text_test)

y_pred = (y_pred > 0.5).astype(int)

y_pred.shape

from sklearn.metrics import multilabel_confusion_matrix
multilabel_confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report

label_names = ['is_offensive', 'INSULT', 'OTHER', 'SEXIST', 'RACIST', 'PROFANITY']

print(classification_report(y_test, y_pred, target_names=label_names))

end_to_end_model = tf.keras.Sequential([
  keras.Input(shape=(1,), dtype="string"),
  vectorize_text,
  model,
  layers.Activation('sigmoid')
])

end_to_end_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="Adam"
)
end_to_end_model.summary()

end_to_end_model.save("end_to_end_model")

loaded_end_to_end_model = tf.keras.models.load_model("end_to_end_model")

def predict(df):
    df = df.copy()
    df['offansive'] = 1
    df['target'] = None
    for i,row in df.iterrows():
      text = row["text"]
      proba = loaded_end_to_end_model.predict([text])
      proba  = np.round(proba,1)
      proba = (proba > 0.5).astype(int)
      offansive_proba = proba[0][0]
      if offansive_proba == 1:
        continue
      else:
        df.at[i, "offansive"] = offansive_proba

      
      if proba[0][1] == 1:
        df.at[i, "target"] = 'INSULT'
      elif proba[0][2] == 1:
        df.at[i, "target"] = 'OTHER'
      elif proba[0][3] == 1:
        df.at[i, "target"] = 'SEXIST'
      elif proba[0][4] == 1:
        df.at[i, "target"] = 'RACIST'
      elif proba[0][5] == 1:
        df.at[i, "target"] = 'PROFANITY'

  
    return df



#Deneme amaçlı test verisi düşünülerek kullanılan bir veriseti 
test_data = pd.read_csv("deneme.csv",sep=";")
test_data

deneme = predict(test_data)

deneme

def gradio_comment(comment):
    
   
    result = loaded_end_to_end_model.predict([comment])
    result  = np.round(result,1)
    result = (result > 0.5).astype(int)
    if result[0][0] == 1:

      if result[0][1] == 1:
        text = "OFFENSİVE/INSULT"
      elif result[0][3] == 1:
        text = 'OFFENSİVE/SEXIST'
      elif result[0][4] == 1:
        text = 'OFFENSİVE/RACIST'
      elif result[0][5] == 1:
        text = 'OFFENSİVE/PROFANITY'
    else:
      text = "NOT OFFENSİVE/OTHER"
    

    return text

GradioGUI = gr.Interface(
    fn=gradio_comment,
    inputs='text',
    outputs='text',
    title='Aşağılayıcı Yorum Tespiti',
    css='''span{text-transform: uppercase} p{text-align: center}''')

GradioGUI.launch(share=True)
