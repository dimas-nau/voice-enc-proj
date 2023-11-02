# Mod encrypt-decrypt function
def encrypt_ez(x, k, n):
    if isinstance(x, int):
        c = (x + k) % n
        return c
    else:
        return x

def decrypt_ez(c, k, n):
    if isinstance(c, int):
        p = (c - k) % n
        return p
    else:
        return c