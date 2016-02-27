#!/bin/bash

# Script for generating x86 assembly code for big integer routines
echo "section .text" > x86.asm

for ((bits=64; bits <=256; bits+=32))
do
    python gen.py add $bits >> x86.asm
    if [ $? -ne 0 ]; then
        echo "Error"
        exit
    fi
    python gen.py sub $bits >> x86.asm
    if [ $? -ne 0 ]; then
        echo "Error"
        exit
    fi
    python gen.py mul $bits >> x86.asm
    if [ $? -ne 0 ]; then
        echo "Error"
        exit
    fi
    python gen.py square $bits >> x86.asm
    if [ $? -ne 0 ]; then
        echo "Error"
        exit
    fi
    python gen.py mul2n $bits >> x86.asm
    if [ $? -ne 0 ]; then
        echo "Error"
        exit
    fi
done
