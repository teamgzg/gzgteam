# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gCt_MGi51OqhfWEUXkgv7SIRpRcOp1Lo
"""

!pip install gradio

import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import gradio as gr

#stop words'ler için
tr_stop_words = pd.read_csv("tr-stop-words.txt",header=None)

#Veri temizleme işlemi
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

#modelin yüklenmesi
loaded_end_to_end_model = tf.keras.models.load_model("end_to_end_model")

def auth(username, password):
    if username == "GZG" and password == "A3ZPAYGJUEWD74NR":
        return True
    else:
        return False


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

def get_file(file):
    output_file = "output_GZG.csv"

    # For windows users, replace path seperator
    file_name = file.name.replace("\\", "/")

    df = pd.read_csv(file_name, sep="|")

    predict(df)
    df.to_csv(output_file, index=False, sep="|")
    return (output_file)


# Launch the interface with user password
iface = gr.Interface(get_file, "file", "file")

if __name__ == "__main__":
    iface.launch(share=True, auth=auth)