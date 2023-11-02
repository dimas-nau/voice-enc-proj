from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from ezencdec import encrypt_ez, decrypt_ez
from syllable_indexer import syl_to_index
import os, glob
import subprocess
import torchaudio
import torch
import numpy as np
import json
import random
from jiwer import cer
import ast
import argparse
import pickle

# as reference files
with open('/path/to/reference.txt', 'r') as syl: # CHANGE PATH LATER
    syllables = syl.readlines()
    
syllables = [line.strip() for line in syllables]
ref_file = '/path/to/reference.txt' # CHANGE PATH LATER


# model dir
repo_name='/path/to/model'

# Loading model and syllableizer
processor = Wav2Vec2Processor.from_pretrained(repo_name)
model_fine = Wav2Vec2ForCTC.from_pretrained(repo_name)

# for random input
vowel = ['a','i','u','e','o']

# Parse argument
parser = argparse.ArgumentParser(description="Audio encryption and decryption script")
parser.add_argument('-n', type=int, help='Number of Characters', required=True)
parser.add_argument('-gen', type=str, help='folder naming, tips: be consistent', required=True)
parser.add_argument('-noise', type=str, help='path for noise folders', required=True)

args = parser.parse_args()
n = args.n # NUMBER OF ITERATION, CHANGED BASED ON DESIRED NUMBER OF CHARACTERS
noise = args.noise  # If you want to do decryption for AWGN audio


t = 5 # total vowel characters
checkpoint_step = 50000 # always remember to change it depends on the chars

# path declaration
gen = args.gen # FOLDER NAMING. TIPS: BE CONSISTENT WITH THIS

# raw_path = '/home/vlsi-elka/Documents/Dimas_folder/model-lain/script/testing/{}/iter_{}/raw_{}/'.format(gen,itid,n)
enc_path = '/path/to/encrypted_audio'.format(gen,noise,n) #ADDED NOISE ON ENCRYPTED
key_path = '/path/to/key'.format(gen,n)
log_path = '/path/to/logs'.format(gen,noise)
dec_path = '/path/to/decrypted_audio'.format(gen,noise,n) # LAST UPDATED

dec_checkpoint_file = os.path.join(log_path, 'dec_checkpoint_{}.pkl')

###############################################################
############### DECRYPTION PROCESS STARTS HERE ################
###############################################################

# Defining encrypted audio
prefix = 'enc_char_'
ext = '.wav'
full_name = os.path.join(enc_path + {noise} + prefix + '*' + ext)
encrypt_target = sorted(glob.glob(full_name), key=lambda x: int(x.split('_')[-1].split('.')[0]))    


# Check if a checkpoint file exists
dec_ckpts = glob.glob(os.path.join(log_path, 'dec_checkpoint_*.pkl'))

if dec_ckpts:
    dec_ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_dec = dec_ckpts[-1]
    # Load the checkpoint data
    with open(latest_dec, 'rb') as file:
        checkpoint_data_dec = pickle.load(file)

    # Retrieve the variables from the checkpoint data
    latest_index = checkpoint_data_dec['latest_index']
    pred_list = checkpoint_data_dec['pred_list']
    pred_index = checkpoint_data_dec['pred_index']
    dec_string = checkpoint_data_dec['dec_string']
    dec_index = checkpoint_data_dec['dec_index']

    encrypt_target = [filename for filename in encrypt_target if int(filename.split('_')[-1].split('.')[0]) > latest_index]
    encrypt_target.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print("Checkpoint found. Resuming decryption from step", latest_index)
    print("RESUMING DECRYPTION PROCESS")
    
else:
    # Initialize variables if no checkpoint file exists
    latest_index = 0
    pred_list = []
    pred_index = []
    dec_string = []
    dec_index = []
    encrypt_target = sorted(glob.glob(full_name), key=lambda x: int(x.split('_')[-1].split('.')[0]))    
    print("STARTING DECRYPTION PROCESS")

# doing decryption for every audio files in encrypted audio folder

for filename in encrypt_target:
    index = filename.split('_')[-1].split('.')[0]
    
    # if int(index) < latest_index:
    #     continue
   
    # Decryption key loading
    with open(os.path.join(key_path, 'enc_key_{}.txt'.format(index)), 'r') as k:
        key_str = k.read()
        key = ast.literal_eval(key_str)
        
    # Audio Processing
    audio_array, sr = torchaudio.load(filename) #load audio file
    resampler = torchaudio.transforms.Resample(sr, 16_000) #resample the audio file
    
    audio_array_resampled = resampler(audio_array).squeeze().numpy()
    inputs = processor(audio_array_resampled, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # Model Prediction
    with torch.no_grad():   
        logits = model_fine(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)
    pred = str(pred[0])
    pred = pred.replace('[PAD]','')
    pred = pred.replace('[UNK]',' ')
    
    # predict result indexing
    pred_idx = syl_to_index(pred, ref_file)
    
    # Decrypt key creation
    length = len(pred_idx)
    decrypt = [decrypt_ez(x, key[k], t) for k, x in enumerate(pred_idx)]
    dec_str = [syllables[val] for val in decrypt]
    
   
    # Result merging
    pred_list.extend(pred)
    pred_index.append(pred_idx)
    dec_string.append(dec_str)
    dec_index.append(decrypt)

    # Save checkpoint after every decryption step
    if (int(index) + 1) % checkpoint_step == 0:
        checkpoint_data_dec = {
            'latest_index': index,
            'pred_list': pred_list,
            'pred_index': pred_index,
            'dec_string': dec_string,
            'dec_index': dec_index
            }
        
        checkpoint_dec = dec_checkpoint_file.format(int(index) + 1)
        # previous_step = (int(index) + 1) - (int(index) + 1) % checkpoint_step
        # previous_dec = dec_checkpoint_file.format(previous_step)
        # if os.path.exists(previous_dec):
        #     print("removing iter_{} previous checkpoint (Decrypt) and writing a new one...".format(itid))
        #     os.remove(previous_dec)
        
        with open(checkpoint_dec, 'wb') as file:
            pickle.dump(checkpoint_data_dec, file)
            print("Checkpoint (Decrypt) saved for {} at step".format(noise), int(index)+1)
    
    # Generating Decrypted audio
    out_dec = f"{dec_path}dec_char_{index}.wav"
    commandes = f"espeak -v mb-id1 {dec_str} -w {out_dec} -s 80 -g 30 -a 90"
    subprocess.call(commandes, shell=True)
    
    if int(index) == n // 2:
        print("{} Decryption is half way complete".format(noise))
    elif int(index) == n-1:
        print("{} said: GG BOIS".format(noise))
        break

# Final checkpoint saving
checkpoint_data_dec = {
    'latest_index': n-1,
    'pred_list': pred_list,
    'pred_index': pred_index,
    'dec_string': dec_string,
    'dec_index': dec_index
}

with open(dec_checkpoint_file.format(n), 'wb') as file:
    pickle.dump(checkpoint_data_dec, file)
print("Final checkpoint (Decrypt) saved for iter_{} at index".format(itid), n)


# Dictionary log
merge_dict = {
        'Prediction'      : pred_list,
        'Pred Index'      : np.concatenate(pred_index).astype(str).tolist(),
        'Decrypted'       : np.concatenate(dec_string).astype(str).tolist(),
        'Decrypt Index'   : np.concatenate(dec_index).astype(str).tolist(),
        # 'Key List'        : np.concatenate(key_list).astype(str).tolist()
    }


# Log writing to txt and json file
with open(log_path + "log_dec_audio_{}.txt".format(noise), "w") as element:
    for crypt_key,value in merge_dict.items():
        element.write('%s:%s\n\n' % (crypt_key,value))
        
with open(log_path + "log_dec_audio_{}.json".format(noise), "w") as element:
    json.dump(merge_dict, element)