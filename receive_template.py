#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:32:53 2023

@author: vlsi-elka
"""

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import json
import torch
import numpy as np

# model dir
repo_name='/path/to/model'

# Loading model and syllableizer
processor = Wav2Vec2Processor.from_pretrained(repo_name)
model_fine = Wav2Vec2ForCTC.from_pretrained(repo_name)


# datas
f = open("/path/to/vocab.json")
syllable = open('/path/to/syllable_file.txt', 'r')
id_syl = syllable.read()
list_syl = id_syl.split('\n')
dict_syl = {v: k for k, v in enumerate(list_syl)}
syllable.close()
vocab = json.load(f)

# audio file path
file_name_c = ('/path/to/audio.wav')
file_name_n = ('/path/to/audio.wav')

# Load audio file
audio_array_c, sr = torchaudio.load(file_name_c)
print(sr)
audio_array_n, sr = torchaudio.load(file_name_n)
resampler = torchaudio.transforms.Resample(sr, 16_000)

# Preprocess the audio array
# Clean
audio_array_resampled_c = resampler(audio_array_c).squeeze().numpy()
inputs_c = processor(audio_array_resampled_c, sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits_c = model_fine(inputs_c.input_values, attention_mask=inputs_c.attention_mask).logits


# Noise
audio_array_resampled_n = resampler(audio_array_n).squeeze().numpy()
inputs_n = processor(audio_array_resampled_n, sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits_n = model_fine(inputs_n.input_values, attention_mask=inputs_n.attention_mask).logits

# Predicting audio array
pred_ids_c = torch.argmax(logits_c, dim=-1)
pred_ids_n = torch.argmax(logits_n, dim=-1)

# Prediction Result
results_c = processor.batch_decode(pred_ids_c)
results_c = str(results_c[0])


results_n = processor.batch_decode(pred_ids_n)
results_n = str(results_n[0])

# result mapping
results_c = results_c.replace('[PAD]', '')
results_n = results_n.replace('[PAD]', '')
results_c = results_c.replace('[UNK]', ' ')
results_n = results_n.replace('[UNK]', ' ')
print("Prediction mic    :", results_c)
print("Prediction direct :", results_n)
print('\n')



# Mod decrypt-decrypt function
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
            with open(input_text, 'r', decoding='utf-8') as f:
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
            # output.append(text[0])
            text = text[1:]
    return output

# Syllable to index function
def syl_to_index(input_text, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        syllables = file.read().splitlines()
        
        if os.path.isfile(input_text):
            with open(input_text, 'r', decoding='utf-8') as f:
                text = f.read()
        else:
            text = input_text
            
    output = []
    while text: #Text input iteration
        for i in range(len(syllables)): #SyllableDict iteration
            if text.startswith(syllables[i]): #Element search in SyllableDict
                output.append(i) #Index adding
                text = text[len(syllables[i]):] #size search
                break 
        else:
            # output.append(text[0])
            text = text[1:]
    return output


# Index to Syllable function
def index_to_syl(input_list, syllables):
    return [syllables[idx] if isinstance(idx, int) else idx for idx in input_list]


# Syllable processing
output_indexed_c = syl_to_index(results_c, 'vocalDict.txt')
output_indexed_n = syl_to_index(results_n, 'vocalDict.txt')

output_syl_c = syl_separator(results_c, 'vocalDict.txt')
output_syl_n = syl_separator(results_n, 'vocalDict.txt')


# Dec list creation
key = 2
n = 5 
list_dec_n = [decrypt_ez(idx, key, n) for idx in output_indexed_n]
list_dec_c = [decrypt_ez(idx, key, n) for idx in output_indexed_c]

list_dec_n_pure = [x for x in list_dec_n if not isinstance(x, str)]
list_dec_c_pure = [x for x in list_dec_c if not isinstance(x, str)]


# Index to String for decrypted and Decrypted indices
str_dec_n = index_to_syl(list_dec_n_pure, list_syl)   

str_dec_c = index_to_syl(list_dec_c_pure, list_syl)

# print result
print('audio str direct  : ', results_n)
print('syllable direct   : ', output_syl_n)
print('list direct       : ', output_indexed_n)
print('dec  direct       : ', list_dec_n)
print('str  direct       : ', str_dec_n)
print('\n')
print('audio str mic     : ', results_c)
print('syllable mic      : ', output_syl_c)
print('list mic          : ', output_indexed_c)
print('dec  mic          : ', list_dec_c)
print('str  mic          : ', str_dec_c)


# Log print to text file
input_dict_clean = {
    'recognized str  ': results_c,
    'syllable        ': output_syl_c,
    'pred index      ': output_indexed_c,
    'dec index       ': list_dec_c,
    'decrypted string': str_dec_c,
    }

input_dict_noise = {
    'recognized str  ': results_n,
    'syllable        ': output_syl_n,
    'pred index      ': output_indexed_n,
    'dec index       ': list_dec_n,
    'decrypted string': str_dec_n,
    }


# Log file writing
with open('rcv_log_file_mic.txt', 'w') as hasil:
    for key, value in input_dict_clean.items():
        hasil.write('%s:%s\n' % (key, value))
        
with open('rcv_log_file_direct.txt', 'w') as hasil:
    for key, value in input_dict_noise.items():
        hasil.write('%s:%s\n' % (key, value))
        
# Text input for eSpeak input
text_file = open("rcv_file_direct_text.txt", "w")
ph = text_file.write(' '.join(output_syl_n))
text_file.close()

text_file = open("rcv_file_mic_text.txt", "w")
ph = text_file.write(' '.join(output_syl_c))
text_file.close()

text_file = open("dec_file_direct.txt", "w")
ph = text_file.write(' '.join(str_dec_n))
text_file.close()

text_file = open("dec_file_mic.txt", "w")
ph = text_file.write(' '.join(str_dec_c))
text_file.close()

# text-to-speech using espeak
from subprocess import call
call(["espeak", "-v", "mb-id1", "-f", "dec_file_mic.txt", "-w", "mic_dec_file.wav", "-s", "90", "-g", "25", "a", "90"])

call(["espeak", "-v", "mb-id1", "-f", "dec_file_direct.txt", "-w", "direct_dec_file.wav", "-s", "90", "-g", "25", "a", "90"])

