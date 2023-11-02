## 🔒 Secure Speech on Phonetic Level 🔒 

This repo consists of programs for research on secure speech on phonetic level utilizing Wav2Vec 2.0 by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli, downloaded from [🤗HuggingfaceHub](https://huggingface.co).

The idea is to simulate a wireless communication from 'transmitter' and 'receiver', by scrambling the information recorded in 'transmitter' and decodes it on the 'receiver' end. To make things as close to reality as possible, we have done the test using several SNR levels by adding white gaussian noise (AWGN). Here are the list of used programs in the conducted research:
1. transmit_receive.py  : The waterfall process from transmit to receive.
2. modded_receive.py    : Modded 'receiver' end to decrypt and receive AWGN audio.
3. cer_checker.py       : Calculate the character error rate (CER) from the prediction process.
4. noise_gen.py         : Add white gaussian noise into audio (done to encrypted audio).
5. decrypt.sh           : A pipeline bash file to do the process in one run, also possible to do parallel processing.
6. train.ipynb          : Notebook code used for model training.

There are several function scripts placed at function folder. The documentation explains the research work flow chart, previous results, and brief explanation of the model.

## How to use
Please note that this process is done in Linux environment, Ubuntu 20.10.
To use the pipeline bash file just simply run this line in your terminal:
```bash
./decrypt.sh
```

The code provided have several arguments:
1. -n: The total number of characters.
2. -gen: Repo name.
3. -seed: Random seed value.
4. -noise: SNR value for decrypt.

## Documentation
The documentation currently only available in Bahasa.
[Document](https://drive.google.com/file/d/17JsUxs567v-eRjiOZFX6_U_PgyXnTrjE/view?usp=sharing)
