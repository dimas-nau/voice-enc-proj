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
with open('/path/to/reference.txt', 'r') as syl:
    syllables = syl.readlines()
    
syllables = [line.strip() for line in syllables]
ref_file = '/path/to/reference.txt'


# model directory
repo_name='/path/to/model'

# Loading model and processor
processor = Wav2Vec2Processor.from_pretrained(repo_name)
model_fine = Wav2Vec2ForCTC.from_pretrained(repo_name)

# vowels list, for random input
vowel = ['a','i','u','e','o']

# Parse argument
parser = argparse.ArgumentParser(description="Audio encryption and decryption script")
parser.add_argument('-n', type=int, help='Number of Characters')
parser.add_argument('-gen', type=str, help='folder naming, tips: be consistent')
parser.add_argument('-i', type=int, help='Iteration Identity (for parallel processing), e.g. iteration-0')
parser.add_argument('-seed', type=int, help='Seed value (for random key generation)')
args = parser.parse_args()
n = args.n # NUMBER OF ITERATION, CHANGED BASED ON DESIRED NUMBER OF CHARACTERS
itid = args.i
seed_value = args.seed
random.seed(seed_value)

t = 5 # total vowel characters
checkpoint_step = 50000 # When the process will save checkpoints,

# path declaration
gen = args.gen # FOLDER NAMING. TIPS: BE CONSISTENT WITH THIS

raw_path = '/path/to/raw_audio'.format(gen,itid,n)
enc_path = '/path/to/encrypted_audio'.format(gen,itid,n)
log_path = '/path/to/log_files'.format(gen,itid)
dec_path = '/path/to/decrypted_audio'.format(gen,itid,n)
enc_checkpoint_file = os.path.join(log_path, 'enc_checkpoint_{}.pkl')
dec_checkpoint_file = os.path.join(log_path, 'dec_checkpoint_{}.pkl')

# if path doesn't exist, it will create new directory
if not os.path.exists(log_path):
    os.makedirs(log_path)
    
if not os.path.exists(raw_path):
    os.makedirs(raw_path)

if not os.path.exists(enc_path):
    os.makedirs(enc_path)

if not os.path.exists(key_path):
    os.makedirs(key_path)

if not os.path.exists(dec_path):
    os.makedirs(dec_path)

#######################################################################
####### RAW AUDIO GENERATION AND ENCRYPTION PROCESS STARTS HERE #######
#######################################################################

# looking for checkpoint pickle file
# Read every checkpoint files
enc_ckpts = glob.glob(os.path.join(log_path, 'enc_checkpoint_*.pkl'))

if enc_ckpts:
    # sort the checkpoint files for every step number
    enc_ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    enc_latest = enc_ckpts[-1]
        
    # Load the checkpoint data
    with open(enc_latest, 'rb') as file:
        checkpoint_data = pickle.load(file)

    # Retrieve the variables from the checkpoint data
    latest_iter = checkpoint_data['latest_iter']
    pred_list = checkpoint_data['pred_list']
    ref_list = checkpoint_data['ref_list']
    pred_index = checkpoint_data['pred_index']
    enc_string = checkpoint_data['enc_string']
    enc_index = checkpoint_data['enc_index']
    key_list = checkpoint_data['key_list']

    print("Checkpoint found. Resuming encryption from step", latest_iter)
    print("RESUMING ENCRYPTION PROCESS...")
    
else:
    # list initialization 
    latest_iter = 0
    pred_list = []
    ref_list = []
    pred_index = []
    enc_string = []
    enc_index = []
    key_list = []
    print("STARTING ENCRYPTION PROCESS")

    # Loop n range, to do encryption every one character generated
for i in range(latest_iter,n):
    out_enc = os.path.join(enc_path, 'enc_char_{}.wav'.format(i))
    out_file = os.path.join(raw_path, 'char_test_{}.wav'.format(i))
    key_file = os.path.join(key_path, 'enc_key_{}.txt'.format(i))
    
    # Raw Audio synthesizing
    char_input = random.choices(vowel)
    espeak_command = f"espeak -v mb-id1 {char_input} -w {out_file} -s 80 -g 30 -a 90"
    subprocess.call(espeak_command, shell=True) # eSpeak Speech Synth
    
      
    # Loading audio data and audio data preprocessing
    audio_files = os.path.join(raw_path, 'char_test_{}.wav'.format(i))
    audio_array, sr = torchaudio.load(audio_files) #load audio file
    resampler = torchaudio.transforms.Resample(sr, 16_000) #resample the audio file
    
    audio_array_resampled = resampler(audio_array).squeeze().numpy()
    inputs = processor(audio_array_resampled, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # Feed to model
    with torch.no_grad():   
        logits = model_fine(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    # Prediction result
    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)
    pred = str(pred[0])
    pred = pred.replace('[PAD]','')
    pred = pred.replace('[UNK]',' ')

    # predict result indexing
    pred_idx = syl_to_index(pred, ref_file)
    
    # Decryption key creation
    length = len(pred_idx)
    key = np.random.randint(0, 5, length)
    encrypt = [encrypt_ez(x, key[k], t) for k, x in enumerate(pred_idx)]
    enc_str = [syllables[val] for val in encrypt]
    
    # Key file writing
    key_step = [x for x in key if not isinstance(x, str)]
    with open (key_file, 'w') as file:
        file.write(str(key_step))
    file.close()
    
    # Result merging in a list
    pred_list.extend(pred)
    ref_list.append(''.join(char_input))
    pred_index.append(pred_idx)
    enc_string.append(enc_str)
    enc_index.append(encrypt)
    key_list.append(key)
    
    # Encrypted audio synthesizing
    commandes = f"espeak -v mb-id1 {enc_str} -w {out_enc} -s 80 -g 30 -a 90"
    subprocess.call(commandes, shell=True)
    # cerpred = cer(char_input, pred)
    # print("CER : ", cerpred)
    
   
    # Checkpoint saving
    if (i + 1) % checkpoint_step == 0:
        checkpoint_data = {
            'latest_iter': i + 1,
            'pred_list': pred_list,
            'ref_list': ref_list,
            'pred_index': pred_index,
            'enc_string': enc_string,
            'enc_index': enc_index,
            'key_list': key_list
            }
        checkpoint_filename = enc_checkpoint_file.format(i + 1)
        # previous_step = (i + 1) - (i + 1) % checkpoint_step
        # previous_check = enc_checkpoint_file.format(previous_step)
        # if os.path.exists(previous_check):
        #     print("removing iter_{} previous checkpoint (Encrypt) and writing a new one...".format(itid))
        #     os.remove(previous_check)
        
        with open(checkpoint_filename, 'wb') as checkpoint_file:
            pickle.dump(checkpoint_data, checkpoint_file)
        print("Checkpoint (Encrypt) saved for iter_{} at step".format(itid), i + 1)        
    
    
    # process mark
    if i == n // 2:
        print("Iteration-{} Encryption half way complete".format(itid))
    
    elif i == n-1:
        print("iteration-{} Encyrption Complete".format(itid))
        break

# Ref and Pred CER calculation
preds = ''.join(pred_list)
refs = ''.join(ref_list)
cerpred = cer(refs, preds)

# Final checkpoint saving
checkpoint_data = {
    'latest_iter': n,
    'pred_list': pred_list,
    'ref_list': ref_list,
    'pred_index': pred_index,
    'enc_string': enc_string,
    'enc_index': enc_index,
    'key_list': key_list
    }

with open(enc_checkpoint_file.format(n), 'wb') as file:
    pickle.dump(checkpoint_data, file)

print("Final checkpoint (Encrypt) saved for iter_{} at step".format(itid), n)


# Dictionary log
merge_dict = {
        'Prediction'      : pred_list,
        'Pred Index'      : np.concatenate(pred_index).astype(str).tolist(),
        'Reference'       : ref_list,
        'Encrypted'       : np.concatenate(enc_string).astype(str).tolist(),
        'Encrypt Index'   : np.concatenate(enc_index).astype(str).tolist(),
        'Key List'        : np.concatenate(key_list).astype(str).tolist()
    }

# Log writing to txt and json file
with open(log_path + "log_audio_iter_{}.txt".format(itid), "w") as element:
    for crypt_key,value in merge_dict.items():
        element.write('%s:%s\n\n' % (crypt_key,value))
        
with open(log_path + "log_audio_iter_{}.json".format(itid), "w") as element:
    json.dump(merge_dict, element)


###############################################################
############### DECRYPTION PROCESS STARTS HERE ################
###############################################################

'''
You can comment this whole section if only you wanted to create Encrypted audio only
'''
# Defining encrypted audio
prefix = 'enc_char_'
ext = '.wav'
full_name = os.path.join(enc_path + prefix + '*' + ext)
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
            print("Checkpoint (Decrypt) saved for iter_{} at step".format(itid), int(index)+1)
    
    # Generating Decrypted audio
    out_dec = f"{dec_path}dec_char_{index}.wav"
    commandes = f"espeak -v mb-id1 {dec_str} -w {out_dec} -s 80 -g 30 -a 90"
    subprocess.call(commandes, shell=True)
    
    if int(index) == n // 2:
        print("Iteration-{} Decryption is half way complete".format(itid))
    elif int(index) == n-1:
        print("iteration-{} said: GG BOIS".format(itid))
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
with open(log_path + "log_dec_audio_iter_{}.txt".format(itid), "w") as element:
    for crypt_key,value in merge_dict.items():
        element.write('%s:%s\n\n' % (crypt_key,value))
        
with open(log_path + "log_dec_audio_iter_{}.json".format(itid), "w") as element:
    json.dump(merge_dict, element)
key_path = '/path/to/keys/key.txt'.format(gen,itid,n)