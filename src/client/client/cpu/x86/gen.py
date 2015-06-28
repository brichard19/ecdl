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
            print("    add eax, dword [ecx + %d]" % (i*4))
        else:
            print("    adc eax, dword [ecx + %d]" % (i*4))
        
        print("    mov dword [edx + %d], eax" % (i*4))
        print("")


    print("    pop ebx")
    print("    mov esp, ebp")
    print("    pop ebp")
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


    print("    pop ebx")
    print("    mov esp, ebp")
    print("    pop ebp")
    print("    ret")
    print("")

# Generate multiplicatin function that computes the lower
# half of the product only
def gen_multiply_low(bits):
    words = bits / 32

    print("global x86_mul_low" + str(bits))
    print("x86_mul_low" + str(bits) + ":")
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


    for i in range(words):
        for j in range(words):
            if i + j < words:
                print("    ; a[%d] * b[%d]" % (i,j))
                if j == 0:
                    print("    mov eax, dword [ecx + %d]" % (i*4))
                    print("    mul dword [esi]")
                    print("    add dword [edi + %d], eax" % ((i+j)*4));
                    print("    adc edx, 0")
                    print("    mov ebx, edx")
                else:
                    print("    mov eax, dword [ecx + %d]" % (i*4))
                    print("    mul dword [esi + %d]" % (j*4))
                    print("    add eax, ebx")
                    print("    adc edx, 0")
                    print("    add dword [edi + %d], eax" % ((i+j)*4))
                    print("    adc edx, 0")
                    print("    mov ebx, edx")

                    print("")
                print("    add dword [edi + %d], ebx" % ((i+words)*4))

    print("    pop esi")
    print("    pop edi")
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


    for i in range(words):
        for j in range(words):
            print("    ; a[%d] * b[%d]" % (i,j))

            if j == 0:
                print("    mov eax, dword [ecx + %d]" % (i*4))
                print("    mul dword [esi + %d]" % (j*4))
                print("    add dword [edi + %d], eax" % ((i+j)*4));
                print("    adc edx, 0")
                print("    mov ebx, edx")
            else:
                print("    mov eax, dword [ecx + %d]" % (i*4))
                print("    mul dword [esi + %d]" % (j*4))
                print("    add eax, ebx")
                print("    adc edx, 0")
                print("    add dword [edi + %d], eax" % ((i+j)*4))
                print("    adc edx, 0")
                print("    mov ebx, edx")

            print("")
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
    elif method == "cmpgte":
        gen_cmpgt(bits)
    elif method == "mul_low":
        gen_multiply_low(bits)
    else:
        print("Invalid method")
        exit()

if __name__ == "__main__":
    main()
