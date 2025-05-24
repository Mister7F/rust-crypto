use crate::aes::aes_constants;
use crate::aes::aes_ni;
use crate::make_array;
use crate::utils::show_progression;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::arch::x86_64::*;

use std::time::SystemTime;

#[inline(always)]
fn _decrypt_round_5(ciphertext: __m128i, key_5: __m128i) -> __m128i {
    aes_ni::xor_si128(ciphertext, key_5)
}

#[inline(always)]
fn _decrypt_round_4(ciphertext: __m128i, key_4: __m128i, key_i: usize) -> u8 {
    let ciphertext = aes_ni::dec_si128(ciphertext, key_4);
    aes_constants::INV_SBOX[aes_ni::get_byte(ciphertext, key_i) as usize]
}

#[pyfunction]
pub fn decrypt_round_5_4(
    ciphertext: [u8; 16],
    key_v: u8,
    key_i: usize,
    column_key_5: [u8; 4],
) -> u8 {
    let col = key_i / 4;
    let mut key_5 = [0u8; 16];
    for i in 0..4 {
        key_5[aes_constants::SR_INV[col * 4 + i] as usize] = column_key_5[i];
    }

    let ciphertext_5 = _decrypt_round_5(aes_ni::load(ciphertext), aes_ni::load(key_5));
    let key_4 = _load_byte(key_v, key_i);
    _decrypt_round_4(ciphertext_5, key_4, key_i)
}

// return the last round key
#[pyfunction]
pub fn aes_5_rounds(ciphertexts: Vec<[[u8; 16]; 256]>, max_column_key: u32) -> [u8; 16] {
    let ciphertexts: Vec<Vec<__m128i>> = ciphertexts
        .iter()
        .map(|cs| cs.iter().map(|c| aes_ni::load(*c)).collect())
        .collect();

    let keys_round_4: Vec<Vec<__m128i>> = (0..=15)
        .map(|key_i| {
            (0..=255)
                .map(|key_v| {
                    let mut buffer = [0u8; 16];
                    buffer[key_i] = key_v;
                    aes_ni::load(buffer)
                })
                .collect()
        })
        .collect();

    // Cache each line as a __m128i, so we can just do 4 xor to build key 5
    // key_5[aes_constants::SR_INV[col * 4 + i as usize] as usize] = column_key[i]
    let key_5_maker = make_array!(16, |position| {
        let key_i = aes_constants::SR_INV[position] as usize;
        make_array!(256, |key_v| _load_byte(key_v as u8, key_i))
    });

    let mut column_keys: [Vec<[u8; 4]>; 4] = Default::default();

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global();

    let now = SystemTime::now();

    for (col, column_key_candidates) in column_keys.iter_mut().enumerate() {
        *column_key_candidates = (0..=max_column_key)
            .into_par_iter()
            .flat_map(|column_key: u32| {
                show_progression(column_key, max_column_key, now);

                // equivalent to C type cast
                let column_key: [u8; 4] = column_key.to_le_bytes();

                let ok = _aes_5_rounds_test_key(
                    col,
                    &ciphertexts,
                    &keys_round_4,
                    &key_5_maker,
                    &column_key,
                );
                if ok {
                    return Some(column_key);
                }
                None
            })
            .collect();

        if column_key_candidates.is_empty() {
            println!("[!] Something went wrong...");
            return [0; 16];
        }
    }

    let total_keys =
        column_keys[0].len() * column_keys[1].len() * column_keys[2].len() * column_keys[3].len();

    println!("[ ] Found {} keys", total_keys);

    // take first key and apply shift row
    let aes_key = column_keys.map(|x| x[0]).concat();
    aes_constants::SR.map(|i| aes_key[i as usize])
}

#[inline(always)]
fn _aes_5_rounds_test_key(
    col: usize,
    ciphertexts: &Vec<Vec<__m128i>>,
    keys_round_4: &Vec<Vec<__m128i>>,
    key_5_maker: &[[__m128i; 256]; 16],
    column_key: &[u8; 4],
) -> bool {
    let delta_sets = ciphertexts.len();

    // load last round key
    let mut key_5 = key_5_maker[col * 4][column_key[0] as usize];
    key_5 = aes_ni::xor_si128(key_5, key_5_maker[col * 4 + 1][column_key[1] as usize]);
    key_5 = aes_ni::xor_si128(key_5, key_5_maker[col * 4 + 2][column_key[2] as usize]);
    key_5 = aes_ni::xor_si128(key_5, key_5_maker[col * 4 + 3][column_key[3] as usize]);

    // pre-decrypt the first delta set (in most case, we will only need
    // that, other delta set are just there to remove more false
    // positive)
    let ciphertexts_5 = make_array!(256, |i| _decrypt_round_5(ciphertexts[0][i], key_5));

    for row in 0..4 {
        let mut row_key_valid = false;
        let key_i = col * 4 + row;

        for key_4 in &keys_round_4[key_i] {
            let mut ok = true;

            for k in 0..delta_sets {
                let mut xor_sum = 0u8;

                for l in 0..256_usize {
                    let ciphertext_5 = if k == 0 {
                        ciphertexts_5[l]
                    } else {
                        _decrypt_round_5(ciphertexts[k][l], key_5)
                    };
                    xor_sum ^= _decrypt_round_4(ciphertext_5, *key_4, key_i);
                }

                if xor_sum != 0 {
                    ok = false;
                    break;
                }
            }
            if ok {
                row_key_valid = true;
                break;
            }
        }
        if !row_key_valid {
            // no byte found for 4th key for this row,
            // "column_key" must be invalid
            return false;
        }
    }

    println!("[ ] New column key found: {:?}", column_key);
    true
}

#[inline(always)]
fn _load_byte(key_v: u8, key_i: usize) -> __m128i {
    let mut buffer = [0u8; 16];
    buffer[key_i] = key_v;
    aes_ni::load(buffer)
}

#[inline(always)]
fn _show_progression(now: SystemTime, column_key: u32, col: u32, max_column_key: u32) {
    if column_key != 0 && column_key % 1000000 == 0 {
        let percent = (column_key * 100) / max_column_key;
        if percent == 0 {
            return;
        }
        let elapsed = now.elapsed().unwrap().as_secs() as u32;

        println!(
            "[ ] Column # {} {}% - {} / {} - {} seconds left",
            col,
            percent,
            column_key,
            max_column_key,
            ((100 - percent) * elapsed) / percent,
        );
    }
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::aes::aes::_AES;
    use crate::aes::aes_5_rounds::aes_5_rounds;
    use crate::make_array;
    use crate::AES;

    #[test]
    fn test_aes_5_rounds() {
        // AES key that produce a last round key with zero
        // at the end of each column to speed up the attack
        let key = [
            244, 85, 188, 201, 131, 81, 37, 153, 89, 151, 0, 59, 224, 39, 99, 226,
        ];

        let cipher = AES!(&key, 5);
        let delta_sets: Vec<[[u8; 16]; 256]> = (0..5)
            .map(|k| {
                make_array!(256, |i| cipher
                    .encrypt([k, k, k, k, k, k, k, k, k, k, k, k, k, k, k, i as u8]))
            })
            .collect::<Vec<_>>();

        let max_column_key = 262144; // 2 ** 18
        let result = aes_5_rounds(delta_sets, max_column_key);
        assert_eq!(
            result,
            [33, 127, 2, 0, 51, 231, 1, 0, 255, 104, 1, 0, 8, 44, 1, 0]
        );
    }

    #[test]
    fn test_aes_5_rounds_flame() {
        // https://github.com/llogiq/flame
        // cargo test --features profile --release
        // AES key that produce a last round key with zero
        // at the end of each column to speed up the attack
        let key = [
            244, 85, 188, 201, 131, 81, 37, 153, 89, 151, 0, 59, 224, 39, 99, 226,
        ];

        let cipher = AES!(&key, 5);
        let delta_sets: Vec<[[u8; 16]; 256]> = (0..5)
            .map(|k| {
                make_array!(256, |i| cipher
                    .encrypt([k, k, k, k, k, k, k, k, k, k, k, k, k, k, k, i as u8]))
            })
            .collect::<Vec<_>>();

        let max_column_key = 262144; // 2 ** 16
        aes_5_rounds(delta_sets, max_column_key);
    }
}
