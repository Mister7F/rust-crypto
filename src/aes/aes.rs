use crate::aes::aes_constants::{sbox, RCON};
use crate::aes::aes_ni;
use crate::utils::{xor_into, ChunkArray};

pub struct _AES {
    round_keys: Vec<[u8; 16]>,
}

impl _AES {
    pub fn new(aes_key: &[u8], rounds: usize) -> Self {
        _AES {
            round_keys: _AES::expand_key(aes_key, rounds),
        }
    }

    pub fn encrypt(&self, plaintext: [u8; 16]) -> [u8; 16] {
        let mut ciphertext = aes_ni::load(plaintext);
        let last_index = self.round_keys.len() - 1;

        ciphertext = aes_ni::xor_si128(ciphertext, aes_ni::load(self.round_keys[0]));

        for round_key in &self.round_keys[1..last_index] {
            let round_key = aes_ni::load(*round_key);
            ciphertext = aes_ni::enc_si128(ciphertext, round_key);
        }

        let round_key = aes_ni::load(self.round_keys[last_index]);
        ciphertext = aes_ni::enc_last_si128(ciphertext, round_key);

        aes_ni::unload(ciphertext)
    }

    fn expand_key(aes_key: &[u8], rounds: usize) -> Vec<[u8; 16]> {
        assert!([16, 24, 32].contains(&aes_key.len()));

        let total_size = (rounds + 1) * 16;
        let mut round_keys: Vec<u8> = aes_key.to_vec();
        round_keys.reserve(total_size);
        let key_size = aes_key.len();

        for k in (key_size..total_size).step_by(4) {
            let mut buffer: [u8; 4] = round_keys[k - 4..].try_into().unwrap();
            if k % key_size == 0 {
                buffer.rotate_left(1);
                sbox(&mut buffer);
                buffer[0] ^= RCON[(k / key_size) % 11];
            }

            if key_size == 32 && k % 32 == 16 {
                // only for AES 256
                sbox(&mut buffer);
            }

            xor_into(&mut buffer, &round_keys[k - key_size..k - key_size + 4]);

            round_keys.extend(buffer);
        }

        round_keys.into_iter().chunk_array().collect()
    }
}

// macro to support default arguments
#[macro_export]
macro_rules! AES {
    ($a: expr, $b: expr) => {
        _AES::new($a, $b)
    };
    ($a: expr) => {
        _AES::new($a, 10)
    };
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::aes::aes::_AES;

    #[test]
    fn test_aes_expand_key_128() {
        // AES 128
        let key = [
            43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60,
        ];
        let result = _AES::expand_key(&key, 2);
        let expected = [
            [
                43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60,
            ],
            [
                160, 250, 254, 23, 136, 84, 44, 177, 35, 163, 57, 57, 42, 108, 118, 5,
            ],
            [
                242, 194, 149, 242, 122, 150, 185, 67, 89, 53, 128, 122, 115, 89, 246, 127,
            ],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_aes_expand_key_192() {
        // AES 192
        let key = [
            36, 240, 61, 145, 0, 164, 199, 100, 176, 80, 195, 138, 199, 72, 62, 87, 63, 199, 55,
            228, 17, 136, 205, 185,
        ];
        let expected = [
            [
                36, 240, 61, 145, 0, 164, 199, 100, 176, 80, 195, 138, 199, 72, 62, 87,
            ],
            [
                63, 199, 55, 228, 17, 136, 205, 185, 225, 77, 107, 19, 225, 233, 172, 119,
            ],
            [
                81, 185, 111, 253, 150, 241, 81, 170, 169, 54, 102, 78, 184, 190, 171, 247,
            ],
        ];
        let result = _AES::expand_key(&key, 2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_aes_expand_key_256() {
        // AES 256
        let key = [
            251, 131, 21, 177, 246, 32, 186, 81, 74, 175, 126, 26, 124, 128, 1, 216, 113, 101, 88,
            127, 217, 74, 254, 201, 72, 107, 139, 213, 142, 211, 241, 255,
        ];
        let expected = [
            [
                251, 131, 21, 177, 246, 32, 186, 81, 74, 175, 126, 26, 124, 128, 1, 216,
            ],
            [
                113, 101, 88, 127, 217, 74, 254, 201, 72, 107, 139, 213, 142, 211, 241, 255,
            ],
            [
                156, 34, 3, 168, 106, 2, 185, 249, 32, 173, 199, 227, 92, 45, 198, 59,
            ],
        ];
        let result = _AES::expand_key(&key, 2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_aes_encrypt() {
        let key = [
            43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60,
        ];
        let ciphertext = [
            34, 73, 162, 99, 140, 111, 28, 117, 90, 132, 249, 104, 26, 159, 8, 193,
        ];
        let plaintext = [
            58, 215, 123, 180, 13, 122, 54, 96, 168, 158, 202, 243, 36, 102, 239, 151,
        ];
        let cipher = AES!(&key);
        assert_eq!(cipher.encrypt(plaintext), ciphertext);

        let ciphertext = [
            223, 160, 180, 148, 240, 251, 72, 13, 226, 20, 46, 172, 19, 162, 47, 239,
        ];
        let cipher = AES!(&key, 5);
        assert_eq!(cipher.encrypt(plaintext), ciphertext);
    }
}
