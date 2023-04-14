# Aşağılayıcı Söylemlerin Doğal Dil İşleme İle Tespiti

Aşağılayıcı Söylem Tespiti için geliştirdiğimiz modelimiz, bize sağlanan veriseti içerisindeki yorumları; veri analizi, veri temizleme, veri önişleme adımlarından geçirip uygun eğitim parametreleri kullanılarak aşağılayıcı yorumları önce `OFFENSİVE(1)` ve `NOT OFFENSİVE(0)` olma durumlarına göre inceler. Daha sonrasında yorum `OFFENSİVE` ise `INSULT`, `SEXIST`, `PROFANITY`,  ve `RACIST` etiketlerinden hangilerine ait olduğunu tahmin etmektedir. Yorum, `NOT OFFENSİVE(0)` ise `OTHER` kabul eder.

- Sunum dosyası için: [Sunum](https://www.canva.com/design/DAFfizWSCN0/ghuc6-K9vScKCZrFHdFpjA/edit?utm_content=DAFfizWSCN0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- Sunum dosyasına pdf olarak ulaşmak için: [Sunum pdf](https://github.com/teamgzg/gzgteam/blob/main/Teknofest_sunum_pdf.pdf)
- Youtube linki için: [Buraya Tıkla!](https://www.youtube.com/watch?v=2ic_A9c6Vh0)

## Takım adı: GZG
Teknofest 2023 Türkçe Doğal Dil İşleme Yarışması için oluşturulmuştur.
#AcikHack2023

## Takım üyeleri:
- Gülzade Evni  **Github:** [GülzadeEvni](https://github.com/GulzadeEvni)
- Zeynep Baydemir  **Github:** [zeynepbaydemir](https://github.com/zeynepbaydemir)

Modelin çalışması için gerekli dosyaları [Hugging Face](https://huggingface.co/TeamGZG/toxic-comment-classification-project) hesabında da bulabilirsiniz.

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
- Türkçe Dolgu Sözcüklerinin (Türkçe stop wordslerin) kaldırılması için `tr-stop-words.txt` dosyası eklenerek aşağıdaki kod satırı çalıştırılmalıdır.

- Repoya eklenmiş olan `tr-stop-words.txt` dosyası indirilerek ekleme işlemi yapılabilir.

`tr-stop-words.txt` dosyasının kullanılacağı kod bloğu:
```shell
tr_stop_words = pd.read_csv("tr-stop-words.txt",header=None)
```

Referans: [Açık Kaynak Turkish Stop Words](https://github.com/ahmetax/trstop) 

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
- Oluşturulan modelin kullanılabilmesi için yüklenen end_to_end modelin içe aktarılmasını sağlayan kod satırı:

```shell 
loaded_end_to_end_model = tf.keras.models.load_model("end_to_end_model") 
```
- end_to_end_model için : [Github](https://github.com/teamgzg/gzgteam/tree/main/end_to_end_model)
- end_to_end_model için : [Hugging Face linki](https://huggingface.co/spaces/TeamGZG/toxic-comment-classificationn)

**Hugging face örnek 1:**

![other1](https://user-images.githubusercontent.com/126960294/229633468-da15bdd9-4e8e-4f61-a2dd-c234415e96c9.png)


**Hugging face örnek 2:**

![sexist1](https://user-images.githubusercontent.com/126960294/229633253-8053399a-66bd-48cb-9891-6a07a57b4850.png)




## Gerekli Fonksiyonların Eklenmesi

- `auth()` fonksiyonu, kullanıcı adı ve şifre ile giriş yapılıp yapılmayacağına karar verir.

- `predict()` fonksiyonu, parametre olarak aldığı DataFrame’e gerekli etiketlemeleri yaparak geriye döndürür.

- `get_file()` fonksiyonu, Gradio servisine yüklenene test verilerini indirerek  pandas dataframe objesine çevirir predict fonksiyonuna gönderir. Predict fonksiyonundan gelen pandas dataframe objesini `output_<takım_ismi>.csv` dosyasına yazar ve Gradio servisi üzerinden dosyayı paylaşır.


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
