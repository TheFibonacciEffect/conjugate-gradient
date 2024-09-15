index_to_cords(int, int, int):
        mov     eax, edi
        mov     edi, edx
        test    edx, edx
        jle     .L2
        xor     ecx, ecx
.L3:
        cdq
        add     ecx, 1
        idiv    esi
        cmp     edi, ecx
        jne     .L3
.L2:
        cdq
        idiv    esi
        mov     eax, edx
        ret
neighbour_index_gpu(int, int, int, int, int, int, int):
        test    esi, esi
        jle     .L7
        xor     r9d, r9d
        mov     r8d, 1
        test    sil, 1
        je      .L8
        mov     r8d, ecx
        mov     r9d, 1
        cmp     esi, 1
        je      .L15
.L8:
        imul    r8d, ecx
        add     r9d, 2
        imul    r8d, ecx
        cmp     esi, r9d
        jne     .L8
.L15:
        imul    edx, r8d
.L7:
        lea     eax, [rdi+rdx]
        ret
laplace_gpu(float*, float*, int, int, int, unsigned int, int):
        push    rbx
        mov     r11d, DWORD PTR [rsp+16]
        cmp     r11d, r8d
        jge     .L16
        movsx   r8, r11d
        mov     eax, ecx
        mov     r10, rdi
        mov     r9d, edx
        lea     rcx, [0+r8*4]
        test    edx, edx
        jle     .L23
        movss   xmm0, DWORD PTR [rsi+r8*4]
        lea     edx, [r11+1]
        pxor    xmm2, xmm2
        mov     rdi, rsi
        movsx   rdx, edx
        movaps  xmm1, xmm0
        addss   xmm1, xmm0
        movaps  xmm0, xmm1
        subss   xmm0, DWORD PTR [rsi+rdx*4]
        subss   xmm0, DWORD PTR [rsi-4+rcx]
        addss   xmm0, xmm2
        cmp     r9d, 1
        je      .L18
        mov     ecx, 1
.L19:
        lea     edx, [rcx-1]
        mov     esi, 1
        mov     ebx, edx
        mov     edx, eax
        and     ebx, 1
        cmp     esi, ecx
        jge     .L40
        test    ebx, ebx
        je      .L20
        mov     esi, 2
        imul    edx, eax
        cmp     esi, ecx
        jge     .L40
.L20:
        imul    edx, eax
        add     esi, 2
        imul    edx, eax
        cmp     esi, ecx
        jl      .L20
.L40:
        add     edx, r11d
        movaps  xmm2, xmm1
        mov     esi, 1
        movsx   rdx, edx
        subss   xmm2, DWORD PTR [rdi+rdx*4]
        lea     edx, [rcx-1]
        mov     ebx, edx
        mov     edx, eax
        and     ebx, 1
        cmp     esi, ecx
        jge     .L41
        test    ebx, ebx
        je      .L21
        mov     esi, 2
        imul    edx, eax
        cmp     esi, ecx
        jge     .L41
.L21:
        imul    edx, eax
        add     esi, 2
        imul    edx, eax
        cmp     esi, ecx
        jl      .L21
.L41:
        mov     esi, r11d
        add     ecx, 1
        sub     esi, edx
        movsx   rdx, esi
        subss   xmm2, DWORD PTR [rdi+rdx*4]
        addss   xmm0, xmm2
        cmp     r9d, ecx
        jne     .L19
.L18:
        movss   DWORD PTR [r10+r8*4], xmm0
.L16:
        pop     rbx
        ret
.L23:
        pxor    xmm0, xmm0
        jmp     .L18