section .text
global x86_add160
x86_add160:

    push ebp
    mov ebp, esp
    push ebx

    mov ebx, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edx, dword [ebp + 16]

    ; a[0] + b[0]
    mov eax, dword [ebx + 0]
    add eax, dword [ecx + 0]
    mov dword [edx + 0], eax

    ; a[1] + b[1]
    mov eax, dword [ebx + 4]
    adc eax, dword [ecx + 4]
    mov dword [edx + 4], eax

    ; a[2] + b[2]
    mov eax, dword [ebx + 8]
    adc eax, dword [ecx + 8]
    mov dword [edx + 8], eax

    ; a[3] + b[3]
    mov eax, dword [ebx + 12]
    adc eax, dword [ecx + 12]
    mov dword [edx + 12], eax

    ; a[4] + b[4]
    mov eax, dword [ebx + 16]
    adc eax, dword [ecx + 16]
    mov dword [edx + 16], eax

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_add320
x86_add320:

    push ebp
    mov ebp, esp
    push ebx

    mov ebx, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edx, dword [ebp + 16]

    ; a[0] + b[0]
    mov eax, dword [ebx + 0]
    add eax, dword [ecx + 0]
    mov dword [edx + 0], eax

    ; a[1] + b[1]
    mov eax, dword [ebx + 4]
    adc eax, dword [ecx + 4]
    mov dword [edx + 4], eax

    ; a[2] + b[2]
    mov eax, dword [ebx + 8]
    adc eax, dword [ecx + 8]
    mov dword [edx + 8], eax

    ; a[3] + b[3]
    mov eax, dword [ebx + 12]
    adc eax, dword [ecx + 12]
    mov dword [edx + 12], eax

    ; a[4] + b[4]
    mov eax, dword [ebx + 16]
    adc eax, dword [ecx + 16]
    mov dword [edx + 16], eax

    ; a[5] + b[5]
    mov eax, dword [ebx + 20]
    adc eax, dword [ecx + 20]
    mov dword [edx + 20], eax

    ; a[6] + b[6]
    mov eax, dword [ebx + 24]
    adc eax, dword [ecx + 24]
    mov dword [edx + 24], eax

    ; a[7] + b[7]
    mov eax, dword [ebx + 28]
    adc eax, dword [ecx + 28]
    mov dword [edx + 28], eax

    ; a[8] + b[8]
    mov eax, dword [ebx + 32]
    adc eax, dword [ecx + 32]
    mov dword [edx + 32], eax

    ; a[9] + b[9]
    mov eax, dword [ebx + 36]
    adc eax, dword [ecx + 36]
    mov dword [edx + 36], eax

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub160
x86_sub160:

    push ebp
    mov ebp, esp
    push ebx

    mov ebx, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edx, dword [ebp + 16]

    ; a[0] + b[0]
    mov eax, dword [ebx + 0]
    sub eax, dword [ecx + 0]
    mov dword [edx + 0], eax

    ; a[1] + b[1]
    mov eax, dword [ebx + 4]
    sbb eax, dword [ecx + 4]
    mov dword [edx + 4], eax

    ; a[2] + b[2]
    mov eax, dword [ebx + 8]
    sbb eax, dword [ecx + 8]
    mov dword [edx + 8], eax

    ; a[3] + b[3]
    mov eax, dword [ebx + 12]
    sbb eax, dword [ecx + 12]
    mov dword [edx + 12], eax

    ; a[4] + b[4]
    mov eax, dword [ebx + 16]
    sbb eax, dword [ecx + 16]
    mov dword [edx + 16], eax

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul160
x86_mul160:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    ; a[0] * b[0]
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add dword [edi + 0], eax
    adc edx, 0
    mov ebx, edx

    ; a[0] * b[1]
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 4], eax
    adc edx, 0
    mov ebx, edx

    ; a[0] * b[2]
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 8], eax
    adc edx, 0
    mov ebx, edx

    ; a[0] * b[3]
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    ; a[0] * b[4]
    mov eax, dword [ecx + 0]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    mov dword [edi + 20], ebx
    ; a[1] * b[0]
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add dword [edi + 4], eax
    adc edx, 0
    mov ebx, edx

    ; a[1] * b[1]
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 8], eax
    adc edx, 0
    mov ebx, edx

    ; a[1] * b[2]
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    ; a[1] * b[3]
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    ; a[1] * b[4]
    mov eax, dword [ecx + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    add dword [edi + 20], eax
    adc edx, 0
    mov ebx, edx

    mov dword [edi + 24], ebx
    ; a[2] * b[0]
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add dword [edi + 8], eax
    adc edx, 0
    mov ebx, edx

    ; a[2] * b[1]
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    ; a[2] * b[2]
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    ; a[2] * b[3]
    mov eax, dword [ecx + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 20], eax
    adc edx, 0
    mov ebx, edx

    ; a[2] * b[4]
    mov eax, dword [ecx + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    add dword [edi + 24], eax
    adc edx, 0
    mov ebx, edx

    mov dword [edi + 28], ebx
    ; a[3] * b[0]
    mov eax, dword [ecx + 12]
    mul dword [esi + 0]
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    ; a[3] * b[1]
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    ; a[3] * b[2]
    mov eax, dword [ecx + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 20], eax
    adc edx, 0
    mov ebx, edx

    ; a[3] * b[3]
    mov eax, dword [ecx + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 24], eax
    adc edx, 0
    mov ebx, edx

    ; a[3] * b[4]
    mov eax, dword [ecx + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    add dword [edi + 28], eax
    adc edx, 0
    mov ebx, edx

    mov dword [edi + 32], ebx
    ; a[4] * b[0]
    mov eax, dword [ecx + 16]
    mul dword [esi + 0]
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    ; a[4] * b[1]
    mov eax, dword [ecx + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 20], eax
    adc edx, 0
    mov ebx, edx

    ; a[4] * b[2]
    mov eax, dword [ecx + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 24], eax
    adc edx, 0
    mov ebx, edx

    ; a[4] * b[3]
    mov eax, dword [ecx + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 28], eax
    adc edx, 0
    mov ebx, edx

    ; a[4] * b[4]
    mov eax, dword [ecx + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    add dword [edi + 32], eax
    adc edx, 0
    mov ebx, edx

    mov dword [edi + 36], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul_low160
x86_mul_low160:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    ; a[0] * b[0]
    mov eax, dword [ecx + 0]
    mul dword [esi]
    add dword [edi + 0], eax
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], ebx
    ; a[0] * b[1]
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 4], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 20], ebx
    ; a[0] * b[2]
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 8], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 20], ebx
    ; a[0] * b[3]
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 20], ebx
    ; a[0] * b[4]
    mov eax, dword [ecx + 0]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 20], ebx
    ; a[1] * b[0]
    mov eax, dword [ecx + 4]
    mul dword [esi]
    add dword [edi + 4], eax
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], ebx
    ; a[1] * b[1]
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 8], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 24], ebx
    ; a[1] * b[2]
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 24], ebx
    ; a[1] * b[3]
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 24], ebx
    ; a[2] * b[0]
    mov eax, dword [ecx + 8]
    mul dword [esi]
    add dword [edi + 8], eax
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], ebx
    ; a[2] * b[1]
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 28], ebx
    ; a[2] * b[2]
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 28], ebx
    ; a[3] * b[0]
    mov eax, dword [ecx + 12]
    mul dword [esi]
    add dword [edi + 12], eax
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], ebx
    ; a[3] * b[1]
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx

    add dword [edi + 32], ebx
    ; a[4] * b[0]
    mov eax, dword [ecx + 16]
    mul dword [esi]
    add dword [edi + 16], eax
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

