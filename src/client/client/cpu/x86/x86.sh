#!/bin/bash

# Script for generating x86 assembly code for big integer routines
echo "section .text" > x86.asm
python gen.py add 160 >> x86.asm
python gen.py add 320 >> x86.asm
python gen.py sub 160 >> x86.asm
python gen.py mul 160 >> x86.asm
python gen.py mul_low 160 >> x86.asm
