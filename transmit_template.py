#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:37:53 2023

@author: Dimas Naufal Wyangputra - 12918014
"""

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import torch
import jiwer
import numpy as np

# model dir
repo_name='/path/to/repo'

# Loading model and syllableizer
processor = Wav2Vec2Processor.from_pretrained(repo_name)
model_fine = Wav2Vec2ForCTC.from_pretrained(repo_name)


# datas
syllable = open('/path/to/syllable/list.txt', 'r')
id_syl = syllable.read()
list_syl = id_syl.split('\n')
dict_syl = {v: k for k, v in enumerate(list_syl)}
syllable.close()
ref_file = '/path/to/ref'

# Load audio file
file_name = ('/path/to/audio.wav')
audio_array, sr = torchaudio.load(file_name)
resampler = torchaudio.transforms.Resample(sr, 16_000)

# Preprocess the audio array
audio_array_resampled = resampler(audio_array).squeeze().numpy()
step = 6000000 #used on large files audio
all_frame_results = np.array(['']*(len(audio_array_resampled/step)+1), dtype=object)

for i in range(0, len(audio_array_resampled), step):
    frame = audio_array_resampled[i:i+step]
    print(i, "to", i+step)
    inputs = processor(frame, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    with torch.no_grad():   
        logits = model_fine(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    results = processor.batch_decode(pred_ids)
    results = str(results[0])
    results = results.replace('[PAD]','')
    results = results.replace('[UNK]',' ')
    all_frame_results[int(i/step)] = results
    print("Prediction:", results)


# Mod encrypt-decrypt function
def encrypt_ez(x, k, n):
    if isinstance(x, int):
        c = (x + k) % n
        return c
    else:
        return x

def decrypt_ez(c, k, n):
    if isinstance(c, int):
        p = (c - k) % n
        return p
    else:
        return c

# String Separator function
def syl_separator(input_text, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        syllables = file.read().splitlines()
        
        if os.path.isfile(input_text):
            with open(input_text, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = input_text
            
    output = []
    while text:
        for syl in syllables:
            if text.startswith(syl):
                output.append(syl)
                text = text[len(syl):]
                break
        else:
            output.append(text[0])
            text = text[1:]
    return output

# Syllable to index function
def syl_to_index(input_text, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        syllables = file.read().splitlines()
        
        if os.path.isfile(input_text):
            with open(input_text, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = input_text
            
    output = []
    while text: #Text input Iteration
        for i in range(len(syllables)): 
            if text.startswith(syllables[i]): #Char searching in syllables
                output.append(i) #Index adding
                text = text[len(syllables[i]):] #Add text search size
                break #menghentikan 
        else:
            output.append(text[0])
            text = text[1:]
    return output


# Index to Syllable function
def index_to_syl(input_list, syllables):
    return [syllables[idx] if isinstance(idx, int) else idx for idx in input_list]

# CER function
def calculate_cer(predictions, targets):
    total_cer = 0
    
    for prediction, target in zip(predictions, targets):
        cer = jiwer.cer(prediction, target)
        total_cer += cer
    avg_cer = total_cer / len(predictions)
    return 1 - avg_cer

results = ''

for frame in all_frame_results:
    results = results+frame

# Syllable processing
output_indexed = syl_to_index(results, ref_file)
output_syl = syl_separator(results, ref_file)

clean_index = [x for x in output_indexed if not isinstance(x, str)]

# Enc-Dec list creation
key = 2
n = 5 #total reference chars
list_enc_c = [encrypt_ez(x, key, n) for x in clean_index]
output_str = index_to_syl(clean_index, list_syl)


# Index to String for Encrypted and Decrypted indices  
str_enc_c = index_to_syl(list_enc_c, list_syl)


# print result
print('audio string   : ', results)
print('syllable       : ', output_syl)
print('list ori       : ', output_indexed)
print('list clean     : ', clean_index)
print('str clean      : ', output_str)
print('enc  clean     : ', list_enc_c)
print('str enc clean  : ', str_enc_c)


# Log print to text file
input_dict = {
    'audio string    ': results,
    'syllable        ': output_syl,
    'list ori        ': output_indexed,
    'list clean      ': clean_index,
    'str clean       ': output_str,
    'enc clean       ': list_enc_c,
    'str enc clean   ': str_enc_c
    }


# Log file writing
with open('tr_log_file.txt', 'w') as hasil:
    for key, value in input_dict.items():
        hasil.write('%s:%s\n' % (key, value))
        
        
# Text input for eSpeak input
text_file = open("tr_file_text.txt", "w")
ph = text_file.write(results)
text_file.close()

text_file = open("tr_file_clean.txt", "w")
ph = text_file.write(' '.join(output_str))
text_file.close()

text_file = open("enc_file_c.txt", "w")
ph = text_file.write(' '.join(str_enc_c))
text_file.close()


# text-to-speech using espeak
from subprocess import call
call(["espeak", "-v", "mb-id1", "-f", "enc_file_c.txt", "-w", "mb_enc_file_c.wav", "-s", "80", "-g", "30", "-a", "90"])
