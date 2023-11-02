# voice-enc-proj
🔒 Secure Speech on Phonetic Level 🔒 


This repo consists of programs for research on secure speech on phonetic level utilizing Wav2Vec 2.0 by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli, downloaded from [🤗HuggingfaceHub](huggingface.co).

The idea is to simulate a wireless communication from 'transmitter' and 'receiver', by scrambling the information before after 'transmitter' and decodes it at 'receiver' end. To make things as close to reality as possible, we done the test using several SNR levels by adding white gaussian noise (AWGN). Here are the list of the code usage:
1. transmit_receive.py  : The waterfall process from transmit to receive.
2. modded_receive.py    : Modded 'receiver' end to decrypt and receive AWGN audio.
3. cer_checker.py       : Calculate the character error rate (CER) from the prediction process.
4. noise_gen.py         : Add white gaussian noise into audio (done to encrypted audio).
5. decrypt.sh           : A pipeline bash file to do the process in one run, also possible to do parallel processing.

There are several function scripts placed at function folder. The documentation explains the research work flow chart, previous results, and brief explanation of the model.
