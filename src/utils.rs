use std::iter::zip;
use std::marker::Copy;
use std::time::SystemTime;

pub fn bytes_from_hex(input: &str) -> Vec<u8> {
    hex::decode(input).expect("Decoding failed")
}

pub fn hex(input: &[u8]) -> String {
    hex::encode(input)
}

pub fn bits_to_u64(bits: impl Iterator<Item = bool>) -> u64 {
    bits.take(64)
        .enumerate()
        .fold(0u64, |acc, (i, b)| acc | ((b as u64) << i))
}

pub fn u64_to_bits(value: u64) -> [bool; 64] {
    let mut ret = [false; 64];
    for (i, r) in ret.iter_mut().enumerate() {
        *r = (value >> i) & 1 != 0;
    }
    ret
}

pub fn xor_into(target: &mut [u8], values: &[u8]) {
    assert_eq!(target.len(), values.len());
    zip(target.iter_mut(), values.iter()).for_each(|(a, b)| *a ^= b);
}

// chunk array
pub struct ChunkArrayIterator<T: ?Sized, const SIZE: usize> {
    inner: T,
}
impl<T: Iterator<Item = U>, U: Copy, const SIZE: usize> Iterator for ChunkArrayIterator<T, SIZE> {
    type Item = [U; SIZE];

    fn next(&mut self) -> Option<[U; SIZE]> {
        let x = self.inner.next()?;
        let mut result = [x; SIZE];
        for i in 1..SIZE {
            result[i] = self.inner.next()?;
        }
        Some(result)
    }
}
pub trait ChunkArray<T> {
    fn chunk_array<const SIZE: usize>(self) -> ChunkArrayIterator<Self, SIZE>;
}
impl<T, U: Iterator<Item = T>> ChunkArray<T> for U {
    fn chunk_array<const SIZE: usize>(self) -> ChunkArrayIterator<Self, SIZE> {
        ChunkArrayIterator { inner: self }
    }
}

#[macro_export]
macro_rules! make_array {
    ($size:expr, $func:expr) => {{
        let mut arr = [$func(0); $size];
        for i in 1..$size {
            arr[i] = $func(i);
        }
        arr
    }};
}

// show the progression when breuteforcing a key
#[inline(always)]
pub fn show_progression(x: u32, max: u32, now: SystemTime) {
    if x % 1000000 != 0 || x == 0 && x < max {
        return;
    }
    let x = x as u128;
    let max = max as u128;

    let percent = (x * 100) / max;
    let elapsed = now.elapsed().unwrap().as_secs() as u128;
    let remaining = (elapsed * (max - x)) / x;
    let total_estimation = (elapsed * max) / x;

    println!(
        "[ ] Progression {}% - {}s / ~{}s - ~{}s left - {} / {}",
        percent, elapsed, total_estimation, remaining, x, max
    );
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::utils::{bits_to_u64, bytes_from_hex, hex, u64_to_bits, xor_into, ChunkArray};

    #[test]
    fn test_bytes_from_hex() {
        assert_eq!(
            bytes_from_hex("aadd1223456789"),
            [170, 221, 18, 35, 69, 103, 137]
        );
    }

    #[test]
    fn test_hex() {
        assert_eq!("aadd1223456789", hex(&[170, 221, 18, 35, 69, 103, 137]));
    }

    #[test]
    fn test_xor_into() {
        let mut source = [170, 221, 18, 35, 69, 103, 137];
        let values = [108, 187, 15, 220, 120, 178, 58];
        let expected = [198, 102, 29, 255, 61, 213, 179];
        xor_into(&mut source, &values);
        assert_eq!(expected, source);
    }

    #[test]
    fn test_chunk_array() {
        let result: Vec<[i32; 2]> = vec![1, 2, 3, 4].into_iter().chunk_array().collect();
        assert_eq!(result, vec![[1, 2], [3, 4]]);

        let result: Vec<[i32; 1]> = vec![1, 2, 3, 4].into_iter().chunk_array().collect();
        assert_eq!(result, vec![[1], [2], [3], [4]]);

        let result: Vec<[i32; 3]> = vec![1, 2, 3, 4].into_iter().chunk_array().collect();
        assert_eq!(result, vec![[1, 2, 3]]);
    }

    #[test]
    fn test_bits_to_u64() {
        let x = 1090501662029120658;
        let x_bits = [
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0,
        ]
        .map(|c| c != 0);
        assert_eq!(bits_to_u64(x_bits.iter().map(|b| *b)), x);
        assert_eq!(u64_to_bits(x), x_bits);
    }
}
