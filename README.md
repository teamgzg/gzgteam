# Aşağılayıcı Söylemlerin Doğal Dil İşleme İle Tespiti

## Takım adı: GZG
Teknofest 2023 Türkçe Doğal Dil İşleme Yarışması için oluşturulmuştur.
#AcikHack2023

## Takım üyeleri:
- Gülzade Evni
- Zeynep Baydemir

> Hazırlanan end_to_end modelin çalışabilmesi için gerekli adımlar aşağıda verilmiştir.

## Gerekli Kütüphaneler
```shell
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import random
import gradio as gr
```

## Stop words dosyasının eklenmesi
Repoya eklenmiş olan `tr-stop-words.txt` dosyası indirilerek ekleme işlemi yapılabilir.

`tr-stop-words.txt` dosyasının kullanılacağı kod bloğu:
```shell
tr_stop_words = pd.read_csv("tr-stop-words.txt",header=None)
```

## Veri Temizleme İşlemi İçin Yapılan Fonksiyonun yüklenmesi
```shell
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
 ```
   
   
## Modelin Yüklenmesi
```shell 
loaded_end_to_end_model = tf.keras.models.load_model("end_to_end_model") 
```
end_to_end_model için : [Hugging Face linki](https://huggingface.co/spaces/TeamGZG/toxic-comment-classificationn)

## Gerekli Fonksiyonların Eklenmesi
```shell 
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
```

## Lisans
Yarışma süresince geliştirilmiş kodlar, veri kümeleri ve diğer bileşenleri GitHub’da erişilebilir biçimde Açık Kaynak Apache lisansı ile [Lisans dosyası](https://github.com/teamgzg/gzgteam/blob/main/LICENSE) adı altında paylaşıyoruz. 
