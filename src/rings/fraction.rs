use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::ops;

#[derive(Debug, Clone)]
pub struct Fraction {
    pub num: BigInt,
    pub den: BigInt,
}

impl Fraction {
    pub fn new(num: BigInt, den: BigInt) -> Self {
        if den.is_zero() {
            panic!("Denominator cannot be zero");
        }

        let g = &num.gcd(&den);
        let num = num / g;
        let den = den / g;

        if den < BigInt::zero() {
            return Self {
                num: -num,
                den: -den,
            };
        }
        Self { num, den }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        let mut nums = s.split("/");
        let num = nums.next().ok_or("No number")?;
        let den = nums.next().unwrap_or("1");

        Ok(Fraction::new(
            BigInt::parse_bytes(num.as_bytes(), 10).ok_or("Invalid number")?,
            BigInt::parse_bytes(den.as_bytes(), 10).ok_or("Invalid number")?,
        ))
    }
}

impl ops::Add for Fraction {
    type Output = Fraction;

    fn add(self, rhs: Fraction) -> Fraction {
        if self.den == rhs.den {
            return Fraction::new(self.num + rhs.num, self.den);
        }

        Fraction::new(
            &self.num * &rhs.den + &rhs.num * &self.den,
            &self.den * &rhs.den,
        )
    }
}

impl ops::Sub for Fraction {
    type Output = Fraction;

    fn sub(self, rhs: Fraction) -> Fraction {
        self + Fraction {
            num: -rhs.num,
            den: rhs.den,
        }
    }
}

impl ops::Div for Fraction {
    type Output = Fraction;

    fn div(self, rhs: Fraction) -> Fraction {
        Fraction::new(self.num * rhs.den, self.den * rhs.num)
    }
}

impl ops::Mul for Fraction {
    type Output = Fraction;

    fn mul(self, rhs: Fraction) -> Fraction {
        Fraction::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl One for Fraction {
    fn one() -> Fraction {
        Fraction::new(BigInt::one(), BigInt::one())
    }
}

impl Zero for Fraction {
    fn zero() -> Fraction {
        Fraction::new(BigInt::zero(), BigInt::one())
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
}

impl Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.is_one() {
            return write!(f, "{}", self.num);
        }
        write!(f, "{} / {}", self.num, self.den)
    }
}

impl PartialEq<Fraction> for Fraction {
    fn eq(&self, rhs: &Fraction) -> bool {
        &self.num * &rhs.den == &rhs.num * &self.den
    }
}

impl PartialEq<i64> for Fraction {
    fn eq(&self, rhs: &i64) -> bool {
        self.num == &self.den * rhs
    }
}

impl PartialOrd<Fraction> for Fraction {
    fn partial_cmp(&self, rhs: &Fraction) -> Option<Ordering> {
        let a = &self.num * &rhs.den;
        let b = &rhs.num * &self.den;
        return a.partial_cmp(&b);
    }
}

impl Eq for Fraction {}
impl Ord for Fraction {
    fn cmp(&self, rhs: &Fraction) -> Ordering {
        return self.partial_cmp(&rhs).unwrap();
    }
}

impl std::iter::Sum<Fraction> for Fraction {
    fn sum<I: Iterator<Item = Fraction>>(iter: I) -> Fraction {
        iter.fold(Fraction::zero(), |acc, f| acc + f)
    }
}
