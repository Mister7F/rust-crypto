use num_bigint::BigUint;
use std::ops;

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Zmod {
    pub value: BigUint,
    pub modulus: Arc<BigUint>,
}

impl Zmod {
    pub fn new(value: BigUint, modulus: Arc<BigUint>) -> Self {
        let value = value % &*modulus;
        Self { value, modulus }
    }
}

impl ops::Add for &Zmod {
    type Output = Zmod;

    fn add(self, rhs: &Zmod) -> Zmod {
        assert_eq!(self.modulus, rhs.modulus, "Modulus mismatch");

        Zmod {
            value: (&self.value + &rhs.value) % &*self.modulus,
            modulus: self.modulus.clone(),
        }
    }
}

impl ops::Sub for &Zmod {
    type Output = Zmod;

    fn sub(self, rhs: &Zmod) -> Zmod {
        assert_eq!(self.modulus, rhs.modulus, "Modulus mismatch");

        Zmod {
            value: (if rhs.value > self.value {
                &self.value + &*self.modulus - &rhs.value
            } else {
                &self.value - &rhs.value
            }) % &*self.modulus,
            modulus: self.modulus.clone(),
        }
    }
}

impl ops::Mul for &Zmod {
    type Output = Zmod;

    fn mul(self, rhs: &Zmod) -> Zmod {
        assert_eq!(self.modulus, rhs.modulus, "Modulus mismatch");

        Zmod {
            value: (&self.value * &rhs.value) % &*self.modulus,
            modulus: self.modulus.clone(),
        }
    }
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zmod() {
        let bi = |s: &str| BigUint::parse_bytes(s.as_bytes(), 10).unwrap();

        let modulus = Arc::new(bi("133333333333333333337"));
        let a = Zmod::new(bi("999999999999999999999999"), modulus.clone());
        assert_eq!(a.value, bi("133333333333333305836"));

        let b = Zmod::new(bi("133333388888999999999"), modulus.clone());
        assert_eq!((&a + &b).value, bi("55555666639161"));

        let b = Zmod::new(bi("133333388888999999999"), modulus.clone());
        assert_eq!((&a - &b).value, bi("133333277777666639174"));

        let b = Zmod::new(bi("133333388888999999999"), modulus.clone());
        assert_eq!((&b - &a).value, bi("55555666694163"));

        let b = Zmod::new(bi("133333388888999999999"), modulus.clone());
        assert_eq!((&a * &b).value, bi("131805496944333461675"));
    }
}
