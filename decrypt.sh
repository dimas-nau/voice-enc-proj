#!/bin/bash

n=100000
gen=gen100000
noise_levels="30 25 20 10"
#noise_levels="20 10"

decrypt_and_cer()
{
    local noise="$1"
    python testing_decrypt_2.py -n "$n" -gen "$gen" -noise "$noise"
    wait
    python cer_decrypt.py -n "$n" -gen "$gen" -noise "$noise"
}

for noise in $noise_levels; do
    decrypt_and_cer "$noise"
done

wait

echo "Process complete, Good luck!"
