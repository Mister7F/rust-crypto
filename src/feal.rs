use crate::utils::show_progression;
use itertools::Itertools;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::time::SystemTime;

#[inline(always)]
fn feal_rotl8(x: u8, n: u8) -> u8 {
    assert!(n <= 8);
    (x << (8 - n)) | (x >> n)
}

#[inline(always)]
fn feal_g_box(x: u8, y: u8, i: u8) -> u8 {
    // allow integer overflow
    feal_rotl8(x.wrapping_add(y).wrapping_add(i), 6)
}

#[inline(always)]
fn feal_f_box(input: u32) -> u32 {
    let [a0, a1, a2, a3] = input.to_be_bytes();

    let mut f1 = a1;
    let mut f2 = a2;
    f1 ^= a0;
    f2 ^= a3;
    f1 = feal_g_box(f1, f2, 1);
    f2 = feal_g_box(f2, f1, 0);
    let f0 = feal_g_box(a0, f1, 0);
    let f3 = feal_g_box(a3, f2, 1);

    u32::from_be_bytes([f0, f1, f2, f3])
}

#[inline(always)]
fn do_round(plaintext: u64, round_key: u32) -> u64 {
    let left = (plaintext >> 32) as u32;
    let right = plaintext as u32;
    let tmp = left ^ feal_f_box(right ^ round_key);
    (right as u64) << 32 | (tmp as u64)
}

#[inline(always)]
fn undo_round(ciphertext: u64, round_key: u32) -> u64 {
    let left = (ciphertext >> 32) as u32;
    let right = (ciphertext & 0xFFFFFFFF) as u32;
    let tmp = right ^ feal_f_box(left ^ round_key);
    (tmp as u64) << 32 | (left as u64)
}

#[inline(always)]
fn swap_32(ciphertext: u64) -> u64 {
    let left = ciphertext >> 32;
    let right = ciphertext & 0xFFFFFFFF;
    (right << 32) | left
}

#[pyfunction]
pub fn feal_encrypt(plain: u64, subkeys: Vec<u32>) -> u64 {
    let rounds = subkeys.len() - 2;

    let mut cipher: u64 = plain;

    cipher ^= (subkeys[rounds] as u64) << 32 | (subkeys[rounds + 1] as u64);

    for round_key in &subkeys[..subkeys.len() - 2] {
        cipher = do_round(cipher, *round_key)
    }
    cipher
}

// break a round when the probability to have the expected difference is 100%
// (we can early break, and we don't need to iterate over all ciphertexts)
fn feal_break_round(
    ciphertexts_pairs: Vec<[u64; 2]>,
    expected_diff: u32,
) -> Result<u32, &'static str> {
    for round_key in 0..=0xFFFFFFFF {
        let mut ok = true;

        for pair in &ciphertexts_pairs {
            let plaintext_0 = undo_round(pair[0], round_key);
            let plaintext_1 = undo_round(pair[1], round_key);

            if ((plaintext_0 ^ plaintext_1) >> 32) as u32 != expected_diff {
                ok = false;
                break;
            }
        }
        if ok {
            return Ok(round_key);
        }
    }
    Err("No key found")
}

// break a round when the probability to have the expected difference is **not** 100%
// (we need to iterate over all ciphertexts to compute a score for each key)
fn _feal_break_round_probabilistic_core(
    ciphertexts_pairs: &Vec<[u64; 2]>,
    expected_diff: u64,
    start: u32,
    end: u32,
) -> (Vec<u32>, usize) {
    let mut max_score = 0;
    let mut result = Vec::<u32>::default();

    let now = SystemTime::now();

    for round_key in start..=end {
        if start == 0 {
            show_progression(round_key, end, now)
        }

        let score = ciphertexts_pairs
            .iter()
            .filter(|[ciphertext_0, ciphertext_1]| {
                let plaintext_0 = undo_round(*ciphertext_0, round_key);
                let plaintext_1 = undo_round(*ciphertext_1, round_key);
                (plaintext_0 ^ plaintext_1) == expected_diff
            })
            .count();

        if score < max_score {
            continue;
        } else if score > max_score {
            println!("[ ] New best key: {:x}, score: {}", round_key, score);
            result.clear();
            max_score = score;
        }

        result.push(round_key);

        if result.len() < 5 {
            println!(
                "[ ] New key: {:x}, all keys: [{}]",
                round_key,
                result.iter().map(|k| format!("0x{:x}", k)).join(", ")
            );
        }
    }

    (result, max_score)
}

// call _feal_break_round_probabilistic_core with many threads
fn feal_break_round_probabilistic(
    ciphertexts_pairs: Vec<[u64; 2]>,
    expected_diff: u64,
) -> Vec<u32> {
    let threads: u32 = 8;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global();

    let thread_range = 0xFFFFFFFF / threads + threads;

    let all_results: Vec<(Vec<u32>, usize)> = (0..threads)
        .into_par_iter()
        .map(move |thread_id| {
            let start = thread_id * thread_range;
            let end = start + thread_range;
            _feal_break_round_probabilistic_core(&ciphertexts_pairs, expected_diff, start, end)
        })
        .collect();

    let max_score = all_results.iter().map(|x| x.1).max().unwrap();
    all_results
        .iter()
        .filter_map(|x| {
            if x.1 == max_score {
                Some(x.0.clone())
            } else {
                None
            }
        })
        .collect::<Vec<Vec<u32>>>()
        .concat()
}

fn collect_ciphertexts(
    n_ciphertexts: u64,
    input_diff: u64,
    encrypt: &dyn Fn(u64) -> u64,
    map: impl Fn(u64) -> u64,
) -> Vec<[u64; 2]> {
    let mut ciphertexts: Vec<[u64; 2]> = Vec::new();
    for plaintext in 1337u64..(1337u64 + n_ciphertexts) {
        let ciphertext_0 = encrypt(plaintext);
        let ciphertext_1 = encrypt(plaintext ^ input_diff);

        ciphertexts.push([map(ciphertext_0), map(ciphertext_1)]);
    }
    ciphertexts
}

fn collect_plaintexts_ciphertexts(
    n_ciphertexts: u64,
    encrypt: &dyn Fn(u64) -> u64,
    map: impl Fn(u64) -> u64,
) -> Vec<[u64; 2]> {
    let mut ciphertexts: Vec<[u64; 2]> = Vec::new();
    for plaintext in 1337u64..(1337u64 + n_ciphertexts) {
        let ciphertext_0 = encrypt(plaintext);

        ciphertexts.push([plaintext, map(ciphertext_0)]);
    }
    ciphertexts
}

fn feal_break_6_rounds(encrypt: &dyn Fn(u64) -> u64) -> Result<[u32; 8], &str> {
    // Break round 6
    let input_diff = 0x0080028A00000202u64;
    let expected_diff = 0x000002020080028au64;
    let ciphertexts = collect_ciphertexts(50, input_diff, encrypt, |c| c);
    let keys_5 = feal_break_round_probabilistic(ciphertexts, expected_diff);

    for key_5 in keys_5 {
        println!("[ ] Key 5: {}", key_5);

        let encrypt_5_rounds = |plaintext| undo_round(encrypt(plaintext), key_5);
        let Ok(result) = feal_break_5_rounds(&encrypt_5_rounds) else {
            continue;
        };

        let mut result = result.to_vec();
        result.insert(5, key_5);
        return Ok(result.try_into().unwrap());
    }

    Err("No key found")
}

#[pyfunction]
#[pyo3(name = "feal_break_6_rounds")]
pub fn py_feal_break_6_rounds(py: Python, encrypt: PyObject) -> PyResult<[u32; 8]> {
    let py_encrypt = |p: u64| {
        encrypt
            .call(py, (p,), None)
            .unwrap()
            .extract::<u64>(py)
            .unwrap()
    };

    let result = feal_break_6_rounds(&py_encrypt);
    if let Ok(result) = result {
        return Ok(result);
    }
    Err(PyTypeError::new_err("Attack failed"))
}

pub fn feal_break_5_rounds(encrypt: &dyn Fn(u64) -> u64) -> Result<[u32; 7], &str> {
    // Break round 5
    let input_diff = 0x0200000080800000u64;
    let expected_diff = 0x2000000u32;
    let ciphertexts = collect_ciphertexts(6, input_diff, encrypt, |c| c);
    let key_4 = feal_break_round(ciphertexts, expected_diff)?;
    println!("[ ] Key 4: {}", key_4);

    // Break round 4
    let undo_round_5 = move |ciphertext: u64| undo_round(ciphertext, key_4);
    let input_diff = 0x8080000000000000u64;
    let expected_diff = 0x2000000u32;
    let ciphertexts = collect_ciphertexts(6, input_diff, encrypt, undo_round_5);
    let key_3 = feal_break_round(ciphertexts, expected_diff)?;
    println!("[ ] Key 3: {}", key_3);

    // Break round 3
    let undo_round_4 = move |ciphertext: u64| undo_round(undo_round_5(ciphertext), key_3);
    let input_diff = 0x8080000000000000u64;
    let expected_diff = 0x80800000u32;
    let ciphertexts = collect_ciphertexts(6, input_diff, encrypt, undo_round_4);
    let key_2 = feal_break_round(ciphertexts, expected_diff)?;
    println!("[ ] Key 2: {}", key_2);

    // Break round 2
    let undo_round_3 = move |ciphertext: u64| undo_round(undo_round_4(ciphertext), key_2);
    let input_diff = 0x00000000123456FFu64;
    let expected_diff = 0x123456FFu32;
    let ciphertexts = collect_ciphertexts(6, input_diff, encrypt, undo_round_3);
    let key_1 = feal_break_round(ciphertexts, expected_diff)?;
    println!("[ ] Key 1: {}", key_1);

    // Break round 1
    // Bruteforce key zero, calculate key 6 7 8 9, and check if it decrypt
    // correctly our ciphertext into our plaintext
    let undo_round_2 = move |ciphertext: u64| undo_round(undo_round_3(ciphertext), key_1);
    let data = collect_plaintexts_ciphertexts(10, encrypt, undo_round_2);

    for key_0 in 0..=0xFFFFFFFF {
        let mut ok = true;
        let key_67 = undo_round(data[0][1], key_0) ^ data[0][0];

        for pair in &data {
            let plaintext = pair[0];
            let ciphertext = pair[1];

            let test_key_67 = undo_round(ciphertext, key_0) ^ plaintext;

            if key_67 != test_key_67 {
                ok = false;
                break;
            }
        }
        if ok {
            println!("[ ] Key 0: {}", key_0);
            println!("[ ] Key 67: {}", key_67);

            let result = [
                key_0,
                key_1,
                key_2,
                key_3,
                key_4,
                (key_67 >> 32) as u32,
                key_67 as u32,
            ];

            // check if the rounds keys are correct
            let ok = (0..100).all(|_| {
                let plaintext: u64 = rand::thread_rng().gen_range(0..u64::MAX);
                feal_encrypt(plaintext, result.to_vec()) == encrypt(plaintext)
            });

            if ok {
                return Ok(result);
            }
        }
    }

    Err("No key found")
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::feal::do_round;
    use crate::feal::feal_break_6_rounds;
    use crate::feal::feal_encrypt;
    use crate::feal::swap_32;
    use crate::feal::undo_round;

    #[test]
    fn test_feal_do_round() {
        let round_key = 0xccddeeff;
        let plaintext = 0xaa11223344556677;
        let ciphertext = do_round(plaintext, round_key);

        assert_eq!(ciphertext, 4923954431438172757);
        assert_eq!(undo_round(ciphertext, round_key), plaintext);
    }

    #[test]
    fn test_feal_encrypt_4_rounds() {
        let subkeys = vec![
            0x001100, 0x223300, 0x445500, 0x667700, 0x8899aabb, 0xccddeeff,
        ];
        let plaintext: u64 = 0xaa11223344556677;
        let ciphertext = feal_encrypt(plaintext, subkeys);
        assert!(ciphertext == 0x453998946e1715c0);

        let ciphertext = undo_round(ciphertext, 0x667700);
        let ciphertext = undo_round(ciphertext, 0x445500);
        let ciphertext = undo_round(ciphertext, 0x223300);
        let ciphertext = undo_round(ciphertext, 0x001100);
        let ciphertext = ciphertext ^ 0x8899aabbccddeeff;
        assert_eq!(ciphertext, plaintext);
    }

    #[test]
    fn test_feal_encrypt_0_rounds() {
        let subkeys = vec![0x00223300, 0x00445500];
        let plaintext: u64 = 0xaa11223344556677;
        let ciphertext = feal_encrypt(plaintext, subkeys);
        let ciphertext = ciphertext ^ (0x00223300 << 32 | 0x00445500);
        assert_eq!(ciphertext, plaintext);
    }

    #[test]
    fn test_feal_encrypt_1_rounds() {
        let subkeys = vec![0x00001100, 0x00223300, 0x00445500];
        let plaintext: u64 = 0xaa11223344556677;
        let ciphertext = feal_encrypt(plaintext, subkeys);
        let ciphertext = undo_round(ciphertext, 0x00001100);
        let ciphertext = ciphertext ^ (0x00223300 << 32 | 0x00445500);
        assert_eq!(ciphertext, plaintext);
    }

    #[test]
    fn test_feal_break_6_rounds() {
        let subkeys = vec![
            0x001100, 0x223300, 0x445500, 0x667700, 0x8899aabb, 0xabcd, 0xa0a0a0a0, 0xb0b0b0b0,
        ];
        let encrypt = |plaintext| feal_encrypt(plaintext, subkeys.clone());

        let result = feal_break_6_rounds(&encrypt).unwrap();
        // It might return round keys that return similar result
        // assert_eq!(result, subkeys);
        let plaintext = 0x1f2e3d4c5b6a7980;
        assert_eq!(encrypt(plaintext), feal_encrypt(plaintext, result.to_vec()));
    }

    #[test]
    fn test_swap_32() {
        assert_eq!(swap_32(0x1122334455667788), 0x5566778811223344);
    }
}
