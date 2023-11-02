import os
import glob
import argparse
import numpy as np
import soundfile as sf
from commpy.channels import awgn

#argument parsing
parser = argparse.ArgumentParser(description="Audio encryption and decryption script")
parser.add_argument('-n', type=int, help='Number of Characters', required=True)
parser.add_argument('-gen', type=str, help='folder naming, tips: be consistent', required=True)
parser.add_argument('-snr', type=int, help='SNR in decibel', required=True)
args = parser.parse_args()
gen = args.gen
n = args.n
snr = args.snr

# path used
enc_path = '/path/to/encrypted/audio'.format(gen, n)
out_path = '/path/to/output_repo'.format(gen,str(snr))

os.path.join(enc_path)
os.path.join(out_path)

# Create path if doesn't exist
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Defining encrypted audio
prefix = 'enc_char_'
ext = '.wav'
full_name = os.path.join(enc_path + prefix + '*' + ext)
encrypt_target = sorted(glob.glob(full_name), key=lambda x: int(x.split('_')[-1].split('.')[0])) 

# Doing iteration process for every audio files inside a folder

for filename in encrypt_target:
    index = filename.split('_')[-1].split('.')[0]
    audio, sr = sf.read(os.path.join(filename))

    # AWGN, the rate parameter is used to imitate the result from MATLAB as close as possible 
    noise = awgn(audio, snr, rate=0.0007) 
    noise_filename = (out_path + '{}db_enc_char_{}.wav').format(str(snr), index)
    sf.write(noise_filename, noise, sr)

    if int(index) == n // 2:
        print("{}db audio generation is halfway complete".format(str(snr)))
    elif int(index) == n-1:
        print("May the noise be with you".format(noise))
        break