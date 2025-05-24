#![allow(dead_code)]

use std::arch::x86_64::*;
use std::mem::transmute;

#[inline(always)]
pub fn load(vector: [u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(vector.as_ptr() as *const _) }
}

#[inline(always)]
pub fn load_vec(value: &Vec<u8>) -> __m128i {
    load(value[..].try_into().unwrap())
}

#[inline(always)]
pub fn unload(vector: __m128i) -> [u8; 16] {
    let mut buffer: [u8; 16] = [0; 16];
    unsafe {
        _mm_storeu_si128(buffer.as_mut_ptr() as *mut __m128i, vector);
    }
    buffer
}

#[inline(always)]
pub fn dec_si128(ciphertext: __m128i, round_key: __m128i) -> __m128i {
    unsafe { _mm_aesdec_si128(ciphertext, round_key) }
}

#[inline(always)]
pub fn enc_si128(plaintext: __m128i, round_key: __m128i) -> __m128i {
    unsafe { _mm_aesenc_si128(plaintext, round_key) }
}

#[inline(always)]
pub fn enc_last_si128(plaintext: __m128i, round_key: __m128i) -> __m128i {
    unsafe { _mm_aesenclast_si128(plaintext, round_key) }
}

#[inline(always)]
pub fn xor_si128(a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_xor_si128(a, b) }
}

#[inline(always)]
pub fn inv_mix_column_si128(a: __m128i) -> __m128i {
    unsafe { _mm_aesimc_si128(a) }
}

#[inline(always)]
pub fn get_byte(a: __m128i, index: usize) -> u8 {
    assert!(index < 16);
    let p: [u8; 16] = unsafe { transmute(a) };
    p[index]
}

// return the previous round key
#[inline(always)]
pub fn single_step_key_inversion<const RCON: i32>(k: __m128i) -> __m128i {
    // K4' = K4 xor K3
    // K3' = K3 xor K2
    // K2' = K2 xor K1
    // K1' = K1 xor SubWord(RotWord(K4')) xor RCON
    let i = unsafe { _mm_slli_si128(k, 4) }; // i = [K3, K2, K1, 0]
    let k = xor_si128(k, i); // k ^= i

    // l = [SubWord(RotWord(K4')) xor RCON, .., .., ..]
    let j = unsafe { _mm_aeskeygenassist_si128(k, RCON) };
    let j = unsafe { _mm_srli_si128(j, 12) }; // l = [0, 0, 0, SubWord(RotWord(K4')) xor RCON]
    xor_si128(k, j) // k ^= j
}
