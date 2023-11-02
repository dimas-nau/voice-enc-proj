import os
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
    while text: #iterasi text input
        for i in range(len(syllables)): #iterasi syllableDict
            if text.startswith(syllables[i]): #mencari syllable dalam file syllableDict
                output.append(i) #menambahkan index pada list output
                text = text[len(syllables[i]):] #menambah ukuran pencarian text
                break #menghentikan 
        else:
            output.append(text[0])
            text = text[1:]
    return output


# Index to Syllable function
def index_to_syl(input_list, syllables):
    return [syllables[idx] if isinstance(idx, int) else idx for idx in input_list]