#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:24:42 2023

@author: Dimas Naufal Wyangputra
"""

from jiwer import cer

# Preserve Vowels
def keep_vowels(string):
    vowels = 'aiueo'
    new_string = ''
    for char in string:
        if char in vowels:
            new_string += char
    return new_string

# Predicted with decrypted (B & B')
# file referensi
with open ('/path/to/original_text/file.txt', 'r') as file:
    referensi = file.read()

# file dekripsi direct
with open ('/path/to/decrypted_text/dec_file.txt', 'r') as file:
    direct_dec = file.read()

# file dekripsi mic
with open ('/path/to/decrypted_text/dec_file.txt', 'r') as file:
    mic_dec = file.read()

# Encrypted with received (A & A')
# file prediksi awal transmit
with open ('/path/to/transmit_text/tr_file_clean.txt', 'r') as file:
    prediksi_transmit = file.read()

# file prediksi awal receive direct
with open ('/path/to/predicted_text/rcv_file_text.txt', 'r') as file:
    prediksi_rc_dir = file.read()

# file prediksi awal receive mic
with open ('/path/to/predicted_text/rcv_file_text.txt', 'r') as file:
    prediksi_rc_mic = file.read()

# file string setelah enkripsi
with open ('/path/to//encrypted_text/enc_file_c.txt', 'r') as file:
    enc_text = file.read()
    
# vowel strict applied
pred_dir = keep_vowels(prediksi_rc_dir)
pred_mic = keep_vowels(prediksi_rc_mic)
ref = keep_vowels(referensi)
dec_dir = keep_vowels(direct_dec)
dec_mic = keep_vowels(mic_dec)
transmit = keep_vowels(prediksi_transmit)
encrypt = keep_vowels(enc_text)

# CER calculation
# Predicted with decrypted
cer_decdir2ref = cer(ref, dec_dir)
cer_decmic2ref = cer(ref, dec_mic)
cer_tr2ref = cer(ref, transmit)
cer_preddir2enc = cer(encrypt, pred_dir)
cer_predmic2enc = cer(encrypt, pred_mic)

print('CER = PREDIKSI dan REFERENSI \n')
print('Dec Direct to ref = ', dec_dir, 'dan', ref)
print('CER : {:.4f}'.format(cer_decdir2ref), '\n')
print('Dec mic to ref = ', dec_mic, ' dan ', ref)
print('CER : {:.4f}'.format(cer_decmic2ref), '\n')
print('Transmit prediction to ref = ', transmit, ' dan ', ref)
print('CER : {:.4f}'.format(cer_tr2ref), '\n')
print('Receive (Direct) Prediction to encrypted string = ', pred_dir, ' dan ', encrypt)
print('CER : {:.4f}'.format(cer_preddir2enc), '\n')
print('Receive (Mic) Prediction to encrypted string = ', pred_mic, ' dan ', encrypt)
print('CER : {:.4f}'.format(cer_predmic2enc), '\n')


result_dict = {}

result_dict['Dec Direct to ref'] = dec_dir, 'dan', ref
result_dict['CER Dec Direct to ref'] = "{:.4f}".format(cer_decdir2ref) + '\n'

result_dict['Dec mic to ref'] = dec_mic, 'dan', ref
result_dict['CER Dec mic to ref'] = "{:.4f}".format(cer_decmic2ref) + '\n'

result_dict['Transmit prediction to ref'] = transmit, 'dan', ref
result_dict['CER Transmit prediction to ref'] = "{:.4f}".format(cer_tr2ref) + '\n'

result_dict['Receive (Direct) Prediction to encrypted string'] = pred_dir, 'dan', encrypt
result_dict['CER Receive (Direct) Prediction to encrypted string'] = "{:.4f}".format(cer_preddir2enc) + '\n'

result_dict['Receive (Mic) Prediction to encrypted string'] = pred_mic, 'dan', encrypt
result_dict['CER Receive (Mic) Prediction to encrypted string'] = "{:.4f}".format(cer_predmic2enc) + '\n'

# print(result_dict)

with open('cer_file.txt', 'w') as log:
    for key, value in result_dict.items():
        log.write('%s:%s\n' % (key, value))

