use rand::prelude::*;
use rand_distr::{Exp, Normal, WeightedIndex};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::read_to_string;
use std::path::Path;

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize)]
pub struct Parameters {
    pub a1: f64,
    pub a2: f64,
    pub b1: f64,
    pub b2: f64,
    pub I: f64,
    pub multiplicity: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
enum Event {
    Death,
    Birth,
    Immigration,
}

impl Parameters {
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Parameters {
        let s = read_to_string(path).expect("Path could not be opened");
        serde_json::from_str(s.as_str()).expect("JSON parsing error in {path}")
    }

    pub fn alpha<N: Into<f64> + Copy>(&self, n: N) -> f64 {
        self.barnu() * self.birth(n) - self.death(n)
    }

    pub fn birth<N: Into<f64>>(&self, n: N) -> f64 {
        self.a1 - self.b1 * n.into()
    }

    pub fn death<N: Into<f64>>(&self, n: N) -> f64 {
        self.a2 + self.b2 * n.into()
    }

    pub fn rate<N: Into<f64> + Copy>(&self, n: N) -> f64 {
        self.I + n.into() * (self.birth(n) + self.death(n))
    }

    pub fn barnu(&self) -> f64 {
        self.multiplicity
            .iter()
            .zip(1..self.multiplicity.len() + 1)
            .map(|(x, y)| x * (y as f64))
            .sum()
    }

    pub fn barnu2(&self) -> f64 {
        let range2 = (1..self.multiplicity.len() + 1).map(|x| x * x);
        self.multiplicity
            .iter()
            .zip(range2)
            .map(|(x, y)| x * (y as f64))
            .sum()
    }

    fn sample_child<R: Rng>(&self, rng: &mut R) -> u32 {
        let wi = WeightedIndex::new(&*self.multiplicity).unwrap();
        (wi.sample(rng) + 1) as u32
    }

    fn sample<R: Rng, N: Into<f64> + Copy>(&self, n: N, rng: &mut R) -> i32 {
        let wi = WeightedIndex::new([self.I, n.into() * self.death(n), n.into() * self.birth(n)])
            .unwrap();
        let events = [Event::Immigration, Event::Death, Event::Birth];
        let event = events[wi.sample(rng)];
        match event {
            Event::Death => -1_i32,
            Event::Immigration => 1_i32,
            Event::Birth => self.sample_child(rng) as i32,
        }
    }
}

impl fmt::Display for Parameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Parameters(a1: {}, a2: {}, b1: {}, b2: {}, I: {})",
            self.a1, self.a2, self.b1, self.b2, self.I
        )
    }
}

pub fn sample_sde_at_time<R: Rng, N: Into<f64>>(
    params: &Parameters,
    n0: N,
    t: f64,
    dt: f64,
    rng: &mut R,
) -> f64 {
    let steps = (t / dt) as u32;
    let rem = t - (steps as f64) * dt;
    let mut n: f64 = n0.into();
    let barnu = params.barnu();
    let barnu2 = params.barnu2();
    let normd = Normal::new(0., f64::sqrt(dt)).unwrap();
    for _ in 0..steps {
        let birth = params.birth(n);
        let death = params.death(n);
        let mu = n * (birth * barnu - death) + params.I;
        let sigma2 = params.I + n * (birth * barnu2 + death);
        let sigma = f64::sqrt(sigma2);
        let dw = normd.sample(rng);
        n += mu * dt + sigma * dw
    }
    if rem > 0. {
        let normd = Normal::new(0., f64::sqrt(rem)).unwrap();
        let birth = params.birth(n);
        let death = params.death(n);
        let mu = n * (birth * barnu - death) + params.I;
        let sigma2 = params.I + n * (birth * barnu2 + death);
        let sigma = f64::sqrt(sigma2);
        let dw = normd.sample(rng);
        n += mu * rem + sigma * dw
    }
    n
}

pub fn sample_branching_at_time<R: Rng>(params: &Parameters, n0: u32, t: f64, rng: &mut R) -> u32 {
    let mut now = 0.0;
    let mut n: i32 = n0 as i32;
    loop {
        let rate = params.rate(n as u32);
        if rate == 0.0 {
            return 0_u32;
        }
        let expd = Exp::new(rate).unwrap();
        let min_time = now + expd.sample(rng);
        if min_time >= t {
            return n as u32;
        }
        // This cannot result in negative values because it decreases by 1 at most and at n=0
        // that's impossible
        n += params.sample(n as u32, rng);
        now = min_time;
    }
}

pub fn sample_at_times<F, N1, N2, N3>(n0: N3, times: &[f64], mut f: F) -> Vec<f64>
where
    F: FnMut(N1, f64) -> N2,
    N1: Into<f64> + Copy,
    N2: Into<N1>,
    N3: Into<N1>,
{
    let mut now = 0.;
    let mut n: N1 = n0.into();
    times
        .iter()
        .map(|x| {
            let dt = x - now;
            now = *x;
            n = f(n, dt).into();
            n.into()
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use rand::prelude::*;

    #[test]
    fn zero_population_branching_no_immigration_is_zero() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1.,
            b2: 1.,
            I: 0.,
            multiplicity: vec![1.],
        };
        let pop = super::sample_branching_at_time(&p, 0, 20., &mut thread_rng());
        assert_eq!(pop, 0);
    }

    #[test]
    fn zero_population_sde_no_immigration_is_zero() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1.,
            b2: 1.,
            I: 0.,
            multiplicity: vec![1.],
        };
        let pop = super::sample_sde_at_time(&p, 0, 20., 1e-4, &mut thread_rng());
        assert_eq!(pop, 0.);
    }

    #[test]
    fn sample_size_matches_times_sde() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1e-8,
            b2: 1e-4,
            I: 400.,
            multiplicity: vec![0.25, 0.75],
        };
        let sde = |x: f64, y| super::sample_sde_at_time(&p, x, y, 1e-4, &mut thread_rng());
        let times = [10., 20., 30., 40., 50.];
        let pop = super::sample_at_times(5000_u32, &times, sde);
        assert_eq!(pop.len(), times.len());
    }

    #[test]
    fn sample_size_matches_times_branching() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1e-8,
            b2: 1e-4,
            I: 400.,
            multiplicity: vec![0.25, 0.75],
        };
        let branch = |x, y| super::sample_branching_at_time(&p, x, y, &mut thread_rng());
        let times = [10., 20., 30., 40., 50.];
        let pop = super::sample_at_times(5000_u32, &times, branch);
        assert_eq!(pop.len(), times.len());
    }
}
