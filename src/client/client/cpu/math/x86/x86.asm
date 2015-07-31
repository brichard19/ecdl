section .text
global x86_add64
x86_add64:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub64
x86_sub64:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul64
x86_mul64:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov dword [edi + 8], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov dword [edi + 12], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square64
x86_square64:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov dword [edi + 8], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov dword [edi + 12], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_add96
x86_add96:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub96
x86_sub96:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul96
x86_mul96:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov dword [edi + 12], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov dword [edi + 16], ebx
    mov ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov dword [edi + 20], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square96
x86_square96:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov dword [edi + 12], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov dword [edi + 16], ebx
    mov ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov dword [edi + 20], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_add128
x86_add128:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub128
x86_sub128:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul128
x86_mul128:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov dword [edi + 16], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov dword [edi + 20], ebx
    mov ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov dword [edi + 24], ebx
    mov ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square128
x86_square128:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov dword [edi + 16], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov dword [edi + 20], ebx
    mov ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov dword [edi + 24], ebx
    mov ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

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

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov dword [edi + 20], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov dword [edi + 24], ebx
    mov ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    mov ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square160
x86_square160:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov dword [edi + 20], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov dword [edi + 24], ebx
    mov ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    mov ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_add192
x86_add192:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub192
x86_sub192:

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

    ; a[5] + b[5]
    mov eax, dword [ebx + 20]
    sbb eax, dword [ecx + 20]
    mov dword [edx + 20], eax

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul192
x86_mul192:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 20]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 20], eax
    adc ebx, 0
    mov dword [edi + 24], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    mov ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    mov ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov dword [edi + 40], ebx
    mov ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov dword [edi + 44], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square192
x86_square192:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 20]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 20], eax
    adc ebx, 0
    mov dword [edi + 24], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    mov ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    mov ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov dword [edi + 40], ebx
    mov ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov dword [edi + 44], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_add224
x86_add224:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub224
x86_sub224:

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

    ; a[5] + b[5]
    mov eax, dword [ebx + 20]
    sbb eax, dword [ecx + 20]
    mov dword [edx + 20], eax

    ; a[6] + b[6]
    mov eax, dword [ebx + 24]
    sbb eax, dword [ecx + 24]
    mov dword [edx + 24], eax

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul224
x86_mul224:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 20]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 24]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    mov ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov dword [edi + 40], ebx
    mov ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov dword [edi + 44], ebx
    mov ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov dword [edi + 48], ebx
    mov ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov dword [edi + 52], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square224
x86_square224:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 20]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 24]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 24], eax
    adc ebx, 0
    mov dword [edi + 28], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    mov ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov dword [edi + 40], ebx
    mov ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov dword [edi + 44], ebx
    mov ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov dword [edi + 48], ebx
    mov ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov dword [edi + 52], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_add256
x86_add256:

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

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_sub256
x86_sub256:

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

    ; a[5] + b[5]
    mov eax, dword [ebx + 20]
    sbb eax, dword [ecx + 20]
    mov dword [edx + 20], eax

    ; a[6] + b[6]
    mov eax, dword [ebx + 24]
    sbb eax, dword [ecx + 24]
    mov dword [edx + 24], eax

    ; a[7] + b[7]
    mov eax, dword [ebx + 28]
    sbb eax, dword [ecx + 28]
    mov dword [edx + 28], eax

    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_mul256
x86_mul256:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov ecx, dword [ebp + 12]
    mov edi, dword [ebp + 16]

    mov ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 20]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 24]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 0]
    mul dword [esi + 28]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 4]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    mov ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 8]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov dword [edi + 40], ebx
    mov ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 12]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov dword [edi + 44], ebx
    mov ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [ecx + 16]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov dword [edi + 48], ebx
    mov ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [ecx + 20]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov dword [edi + 52], ebx
    mov ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov eax, dword [ecx + 24]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 52], eax
    adc ebx, 0
    mov dword [edi + 56], ebx
    mov ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 52], eax
    adc ebx, 0
    mov eax, dword [ecx + 28]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 56], eax
    adc ebx, 0
    mov dword [edi + 60], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

global x86_square256
x86_square256:

    push ebp
    mov ebp, esp
    push ebx
    push edi
    push esi

    mov esi, dword [ebp + 8]
    mov edi, dword [ebp + 12]

    mov ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 0]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 0], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 4]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 8]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 12]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 16]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 20]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 24]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 0]
    mul dword [esi + 28]
    add eax, ebx
    mov ebx, edx
    mov dword [edi + 28], eax
    adc ebx, 0
    mov dword [edi + 32], ebx
    mov ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 4], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 4]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov dword [edi + 36], ebx
    mov ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 8], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 8]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov dword [edi + 40], ebx
    mov ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 12], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 12]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov dword [edi + 44], ebx
    mov ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 16], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [esi + 16]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov dword [edi + 48], ebx
    mov ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 20], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [esi + 20]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov dword [edi + 52], ebx
    mov ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 24], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov eax, dword [esi + 24]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 52], eax
    adc ebx, 0
    mov dword [edi + 56], ebx
    mov ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 0]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 28], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 4]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 32], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 8]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 36], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 12]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 40], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 16]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 44], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 20]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 48], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 24]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 52], eax
    adc ebx, 0
    mov eax, dword [esi + 28]
    mul dword [esi + 28]
    add eax, ebx
    adc edx, 0
    mov ebx, edx
    add dword [edi + 56], eax
    adc ebx, 0
    mov dword [edi + 60], ebx
    pop esi
    pop edi
    pop ebx
    mov esp, ebp
    pop ebp
    ret

