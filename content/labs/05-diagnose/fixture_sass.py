"""Sample compiler output for the lab's parsing exercises.

These strings are written by hand in the format `cuobjdump -sass` and
`nvcc -Xptxas -v` print. They are teaching samples, not the output of a
particular compile.
"""

PTXAS_CLEAN = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z6kernelPKfPfi' for 'sm_80'
ptxas info    : Function properties for _Z6kernelPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, 380 bytes cmem[0]
"""

PTXAS_SPILL = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z5tiledPKfPfi' for 'sm_80'
ptxas info    : Function properties for _Z5tiledPKfPfi
    64 bytes stack frame, 48 bytes spill stores, 52 bytes spill loads
ptxas info    : Used 255 registers, 8192 bytes smem, 380 bytes cmem[0]
"""

SASS_SCALAR = """\
        /*0000*/                   MOV R1, c[0x0][0x28] ;                    /* 0x00000a0000017a02 */
                                                                             /* 0x000fd00000000f00 */
        /*0010*/                   S2R R0, SR_CTAID.X ;                      /* 0x0000000000007919 */
                                                                             /* 0x000e220000002500 */
        /*0020*/                   S2R R3, SR_TID.X ;
        /*0030*/                   IMAD R0, R0, 0x100, R3 ;
        /*0040*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT ;
        /*0050*/               @P0 EXIT ;
        /*0060*/                   IMAD.WIDE R2, R0, 0x4, c[0x0][0x160] ;
        /*0070*/                   LDG.E R2, [R2.64] ;
        /*0080*/                   FFMA R7, R2, 2, 1 ;
        /*0090*/                   IMAD.WIDE R4, R0, 0x4, c[0x0][0x168] ;
        /*00a0*/                   STG.E [R4.64], R7 ;
        /*00b0*/                   EXIT ;
        /*00c0*/                   BRA 0xc0 ;
"""

SASS_VECTOR = """\
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_CTAID.X ;
        /*0020*/                   S2R R3, SR_TID.X ;
        /*0030*/                   IMAD R0, R0, 0x100, R3 ;
        /*0040*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT ;
        /*0050*/               @P0 EXIT ;
        /*0060*/                   IMAD.WIDE R2, R0, 0x10, c[0x0][0x160] ;
        /*0070*/                   LDG.E.128 R4, [R2.64] ;
        /*0080*/                   FFMA R4, R4, 2, 1 ;
        /*0090*/                   FFMA R5, R5, 2, 1 ;
        /*00a0*/                   FFMA R6, R6, 2, 1 ;
        /*00b0*/                   FFMA R7, R7, 2, 1 ;
        /*00c0*/                   IMAD.WIDE R2, R0, 0x10, c[0x0][0x168] ;
        /*00d0*/                   STG.E.128 [R2.64], R4 ;
        /*00e0*/                   BAR.SYNC 0x0 ;
        /*00f0*/                   HMMA.16816.F32.BF16 R8, R4, R12, R8 ;
        /*0100*/              @!P1 BRA 0x120 ;
        /*0110*/                   LDG.E.64 R10, [R2.64] ;
        /*0120*/                   EXIT ;
"""
