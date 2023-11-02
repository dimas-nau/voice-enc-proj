#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from jiwer import cer
import argparse
import json
import os

# Parse argument
parser = argparse.ArgumentParser(description="Audio encryption and decryption script")
parser.add_argument('-n', type=int, help='Number of Characters')
parser.add_argument('-gen', type=str, help='folder naming, tips: be consistent')
parser.add_argument('-noise', type=str, help='SNR used')

args = parser.parse_args()
n = args.n # NUMBER OF ITERATION, CHANGED BASED ON DESIRED NUMBER OF CHARACTERS
gen = args.gen # FOLDER NAMING. TIPS: BE CONSISTENT WITH THIS
noise = args.noise

# Path to log files
enc_logs = '/path/to/enc_log.json'.format(gen, n)
dec_logs = '/path/to/dec_log.json'.format(gen,noise)
log_noise = '/path/to/log_after_calculation'.format(gen)

with open(enc_logs, 'r') as file:
    tr_log = json.load(file)

with open(dec_logs, 'r') as file:
    rc_log = json.load(file)
    

referensi = ''.join(tr_log['Reference'])
enc_text = ''.join(tr_log['Encrypted'])
receive = ''.join(rc_log['Prediction'])
dec_dir = ''.join(rc_log['Decrypted'])

cer_dec2ref = cer(referensi, dec_dir)
cer_enc2rc = cer(enc_text, receive)


result_dict = {}

result_dict['Ref to Dec'] = referensi, 'dan', dec_dir
result_dict['CER Ref to Dec'] = "{:.6f}".format(cer_dec2ref) +'\n'

result_dict['Encrypt to Receive'] = enc_text, 'dan', receive
result_dict['CER Encrypt to Receive'] = "{:.6f}".format(cer_enc2rc) + '\n'

with open(os.path.join(log_noise, 'cer_dec_{}_{}db.txt'.format(n, noise)), 'w') as log:
    for key, value in result_dict.items():
        log.write('%s:%s\n' % (key, value))
