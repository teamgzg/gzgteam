# gzgteam
Teknofest 2023 Türkçe Doğal Dil İşleme #AcikHack2023

#Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import random
import gradio as gr

#Stop words dosyasının eklenmesi
Repoya eklenmiş olan tr-stop-words.txt dosyası indirilerek ekleme işlemi yapılabilir.

#Veri Temizleme İşlemi İçin Yapılan Fonksiyonun yüklenmesi
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
    
Modelin Yüklenmesi
loaded_end_to_end_model = tf.keras.models.load_model("end_to_end_model")
end_to_end_model için : 
