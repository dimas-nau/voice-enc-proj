# XOR encrypt-decrypt function
def encrypt_x(x, n):
    flag = True #definisi nilai flag
    
    try:
        int(x) #percobaan encrypt jika x adalah integer
    except ValueError:
        flag = False #jika gagal ternary operator return nilai x != integer
    
    return x ^ n if flag else x

def decrypt_x(x, n):
    
    flag = True #definisi nilai flag
    
    try:
        int(x) #percobaan decrypt jika x adalah integer
    except ValueError:
        flag = False #jika gagal ternary operator return nilai x
    
    return x ^ n if flag else x