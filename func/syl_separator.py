import os
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