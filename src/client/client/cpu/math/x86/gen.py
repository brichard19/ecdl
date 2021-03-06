'''
This program generates x86 assembly code for big integer arithmetic
'''

import sys

# Generate addition function
def gen_add(bits):
    words = bits / 32

    print("global x86_add" + str(bits))
    print("x86_add" + str(bits) + ":")
    print("")
    print("push ebx")

    print("    mov ebx, dword [esp + 8]")
    print("    mov ecx, dword [esp + 12]")
    print("    mov edx, dword [esp + 16]")
    print("")

    print("    mov eax, dword [ebx]")
    print("    add eax, dword [ecx]")
    print("    mov dword [edx], eax")
   
    for i in range(1,words):
        print("    mov eax, dword [ebx + %d]" % (i*4))
        print("    adc eax, dword [ecx + %d]" % (i*4))
        
        print("    mov dword [edx + %d], eax" % (i*4))
        print("")

    print("    pop ebx")
    print("    ret")
    print("")

# Generate subtraction function
def gen_sub(bits):
    words = bits / 32

    print("global x86_sub" + str(bits))
    print("x86_sub" + str(bits) + ":")
    print("")
    print("    push ebp")
    print("    mov ebp, esp")
    print("    push ebx")
    print("")
   
    print("    mov ebx, dword [ebp + 8]")
    print("    mov ecx, dword [ebp + 12]")
    print("    mov edx, dword [ebp + 16]")
    print("")

    for i in range(words):
        print("    ; a[%d] + b[%d]" % (i,i))
        print("    mov eax, dword [ebx + %d]" % (i*4))
        if i == 0:
            print("    sub eax, dword [ecx + %d]" % (i*4))
        else:
            print("    sbb eax, dword [ecx + %d]" % (i*4))
        
        print("    mov dword [edx + %d], eax" % (i*4))
        print("")

    # Return the borrow bit
    print("    mov eax, 0")
    print("    adc eax, 0")

    print("    pop ebx")
    print("    mov esp, ebp")
    print("    pop ebp")
    print("    ret")
    print("")

def gen_multiply(bits):
    words = bits / 32

    print("global x86_mul" + str(bits))
    print("x86_mul" + str(bits) + ":")
    print("")
    print("    push ebp")
    print("    mov ebp, esp")
    print("    push ebx")
    print("    push edi")
    print("    push esi")
    print("")
   
    print("    mov esi, dword [ebp + 8]")
    print("    mov ecx, dword [ebp + 12]")
    print("    mov edi, dword [ebp + 16]")
    print("")


    # Do b[0] * a[0] to a[n-1]. This is a separate loop because we move the
    # result to the buffer, not add them
    print("    mov ebx, 0")
    for j in range(0, words):
        print("    mov eax, dword [ecx + %d]" % (0))
        print("    mul dword [esi + %d]" % (j*4))
        print("    add eax, ebx")
        print("    mov ebx, edx")
        print("    mov dword [edi + %d], eax" % (j*4))
        print("    adc ebx, 0")

    print("    mov dword [edi + %d], ebx" % (words*4))

    
    for i in range(1, words):

        #Reset high word to 0
        print("    mov ebx, 0")

        for j in range(words):

            # Read b[i] from memory
            print("    mov eax, dword [ecx + %d]" % (i*4))

            # Multiply by a[j]
            print("    mul dword [esi + %d]" % (j*4))

            # add old high to new low word
            print("    add eax, ebx")

            # add the carry
            print("    adc edx, 0")

            # store new high word in ebx
            print("    mov ebx, edx")

            # add new low + old high to s[i + j]
            print("    add dword [edi + %d], eax" % ((i+j)*4))

            # add the carry to the new high word
            print("    adc ebx, 0")

        # Write high word to the end
        print("    mov dword [edi + %d], ebx" % ((i+words)*4))
    

    print("    pop esi")
    print("    pop edi")
    print("    pop ebx")
    print("    mov esp, ebp")
    print("    pop ebp")
    print("    ret")
    print("")

# Performs N by 2N multiplication
def gen_multiply2n(bits):
    #words = bits / 32
    n1Bits = bits
    n1Words = n1Bits / 32

    n2Bits = 2 * bits
    n2Words = n2Bits / 32

    print("global x86_mul" + str(n1Bits) + "_" + str(n2Bits))
    print("x86_mul" + str(n1Bits) + "_" + str(n2Bits) + ":");
    print("")
    print("    push ebp")
    print("    mov ebp, esp")
    print("    push ebx")
    print("    push edi")
    print("    push esi")
    print("")
   
    #print("    mov esi, dword [ebp + 8]")
   #print("    mov ecx, dword [ebp + 12]")
    print("    mov ecx, dword [ebp + 8]")
    print("    mov esi, dword [ebp + 12]")
    print("    mov edi, dword [ebp + 16]")
    print("")


    # Do b[0] * a[0] to a[n-1]. This is a separate loop because we move the
    # result to the buffer, not add them
    print("    mov ebx, 0")
    for j in range(0, n2Words):
        print("    mov eax, dword [ecx + %d]" % (0))
        print("    mul dword [esi + %d]" % (j*4))
        print("    add eax, ebx")
        print("    mov ebx, edx")
        print("    mov dword [edi + %d], eax" % (j*4))
        print("    adc ebx, 0")

    print("    mov dword [edi + %d], ebx" % (n2Words*4))

    
    for i in range(1, n1Words):

        #Reset high word to 0
        print("    mov ebx, 0")

        for j in range(n2Words):

            # Read b[i] from memory
            print("    mov eax, dword [ecx + %d]" % (i*4))

            # Multiply by a[j]
            print("    mul dword [esi + %d]" % (j*4))

            # add old high to new low word
            print("    add eax, ebx")

            # add the carry
            print("    adc edx, 0")

            # store new high word in ebx
            print("    mov ebx, edx")

            # add new low + old high to s[i + j]
            print("    add dword [edi + %d], eax" % ((i+j)*4))

            # add the carry to the new high word
            print("    adc ebx, 0")

        # Write high word to the end
        print("    mov dword [edi + %d], ebx" % ((i+n2Words)*4))
    

    print("    pop esi")
    print("    pop edi")
    print("    pop ebx")
    print("    mov esp, ebp")
    print("    pop ebp")
    print("    ret")
    print("")

def gen_square(bits):
    words = bits / 32

    print("global x86_square" + str(bits))
    print("x86_square" + str(bits) + ":")
    print("")
    print("    push ebp")
    print("    mov ebp, esp")
    print("    push ebx")
    print("    push edi")
    print("    push esi")
    print("")
   
    print("    mov esi, dword [ebp + 8]")
    print("    mov edi, dword [ebp + 12]")
    print("")


    # Do b[0] * a[0] to a[n-1]. This is a separate loop because we move the
    # result to the buffer, not add them
    print("    mov ebx, 0")
    for j in range(0, words):
        print("    mov eax, dword [esi + %d]" % (0))
        print("    mul dword [esi + %d]" % (j*4))
        print("    add eax, ebx")
        print("    mov ebx, edx")
        print("    mov dword [edi + %d], eax" % (j*4))
        print("    adc ebx, 0")

    print("    mov dword [edi + %d], ebx" % (words*4))

    
    for i in range(1, words):

        #Reset high word to 0
        print("    mov ebx, 0")

        for j in range(words):

            # Read b[i] from memory
            print("    mov eax, dword [esi + %d]" % (i*4))

            # Multiply by a[j]
            print("    mul dword [esi + %d]" % (j*4))

            # add old high to new low word
            print("    add eax, ebx")

            # add the carry
            print("    adc edx, 0")

            # store new high word in ebx
            print("    mov ebx, edx")

            # add new low + old high to to s[i + j]
            print("    add dword [edi + %d], eax" % ((i+j)*4))

            # add the carry to the new high word
            print("    adc ebx, 0")

        # Write high word to the end
        print("    mov dword [edi + %d], ebx" % ((i+words)*4))
    

    print("    pop esi")
    print("    pop edi")
    print("    pop ebx")
    print("    mov esp, ebp")
    print("    pop ebp")
    print("    ret")
    print("")

# Generate multiplication function
def main():
    if len(sys.argv) < 3:
        print("Invalid arguments")
        exit()
   
    method = sys.argv[1]
    bits = int(sys.argv[2])

    if method == "add":
        gen_add(bits)
    elif method == "sub":
        gen_sub(bits)
    elif method == "mul":
        gen_multiply(bits)
    elif method == "square":
        gen_square(bits)
    elif method == "cmpgte":
        gen_cmpgt(bits)
    elif method == "mul2n":
        gen_multiply2n(bits)
    else:
        print("Invalid method")
        exit(1)

if __name__ == "__main__":
    main()
