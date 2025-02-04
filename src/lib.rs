use rand::distr::weighted::WeightedIndex;
use rand::Rng;
use rand_distr::{Distribution, Exp, Normal};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::read_to_string;
use std::path::Path;

/// System parameters that define the rates of birth, death and immigration.
///
/// Birth is defined as $\text{birth}(n) = a_1n - b_1n^2$.
///
/// Death is defined as $\text{death}(n) = a_2n + b_2n^2$
///
/// Thus a1 and a2 are the usual per-individual birth and death rates, and b1 and b2 are the
/// feedback effects to first order of increasing the population size.
/// a1 and a2 have units of 1 per second, and b1 and b2 have units of 1 per second-individual
///
/// I is the immigration rates, in individuals per second (or 1 per second, really).
///
/// The multiplicity is used as a weight vector for the litter size (how many individuals are
/// born in a single birth event). The first value is the weight for a single new individual,
/// the second for 2 new individuals, and so on. One could normalize these so they sum up to 1
/// but that's not necessary. The units are just \[1\].
#[allow(non_snake_case)]
#[derive(Serialize, Deserialize, Debug)]
pub struct Parameters {
    /// Per individual birth rate
    pub a1: f64,
    /// Per individual death rate
    pub a2: f64,
    /// Birth negative feedback coefficient
    pub b1: f64,
    /// Death feedback coefficient
    pub b2: f64,
    /// Immigration rate
    pub I: f64,
    /// Multiplicity weight vector
    pub multiplicity: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
enum Event {
    Death,
    Birth,
    Immigration,
}

impl Parameters {
    /// This function opens the file at the given path and creates parameters from the JSON
    /// string written in the file.
    /// We should provide a method to create such JSON files in a script for ease of use.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Parameters {
        let s = read_to_string(path).expect("Path could not be opened");
        serde_json::from_str(s.as_str()).expect("JSON parsing error in {path}")
    }

    /// This function creates the parameters from a single slice of floats, in the usual order.
    /// This is used for CLI interface creation of Parameters in our application.
    pub fn from_vec(v: &[f64]) -> Parameters {
        Parameters {
            a1: v[0],
            a2: v[1],
            b1: v[2],
            b2: v[3],
            I: v[4],
            multiplicity: v[5..].to_vec(),
        }
    }

    /// alpha is the "low population" exponential time dependence of the system. When positive,
    /// the system grows exponentially with this rate, when negative decays exponentially with
    /// this rate. Since we may have feedback effects through b1 and b2, a growing population
    /// could reach an asymptotic stable size eventually.
    pub fn alpha<N: Into<f64> + Copy>(&self, n: N) -> f64 {
        self.barnu() * self.birth(n) - self.death(n)
    }

    /// The per-second birth rate at the current population size.
    pub fn birth<N: Into<f64>>(&self, n: N) -> f64 {
        self.a1 - self.b1 * n.into()
    }

    /// The per-second death rate at the current population size.
    pub fn death<N: Into<f64>>(&self, n: N) -> f64 {
        self.a2 + self.b2 * n.into()
    }

    /// The per-second total rate of any occurence at the current population size.
    pub fn rate<N: Into<f64> + Copy>(&self, n: N) -> f64 {
        self.I + n.into() * (self.birth(n) + self.death(n))
    }

    /// The average litter size for this system. This can and should be computed once for every
    /// system and then reused.
    pub fn barnu(&self) -> f64 {
        let total: f64 = self.multiplicity.iter().sum::<f64>();
        self.multiplicity
            .iter()
            .zip(1..self.multiplicity.len() + 1)
            .map(|(x, y)| x * (y as f64))
            .sum::<f64>()
            / total
    }

    /// The second moment (mean x^2 value) of the multiplicity vector. This can and should be
    /// computed once for every system and then reused.
    pub fn barnu2(&self) -> f64 {
        let total: f64 = self.multiplicity.iter().sum::<f64>();
        let range2 = (1..self.multiplicity.len() + 1).map(|x| x * x);
        self.multiplicity
            .iter()
            .zip(range2)
            .map(|(x, y)| x * (y as f64))
            .sum::<f64>()
            / total
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

/// Sample an SDE model with forward Euler numeric scheme for a given system to a given time.
///
/// # Arguments
///
/// * `params` - System parameters.
/// * `n0` - Initial population size.
/// * `t` - Time to sample the population size at.
/// * `dt` - Maximal step size for the forward Euler steps.
/// * `rng` - Random number generator to use.
///
/// # Examples
///
/// ```
/// use popfeedback::{Parameters, sample_sde_at_time};
/// use rand::rng;
/// let par = Parameters {a1: 1., a2: 1., b1: 0., b2: 0., I: 500., multiplicity: vec![0.5, 0.5]};
/// sample_sde_at_time(&par, 1000, 10., 1e-3, &mut rng());
///
/// ```
///
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

/// Sample a branching model for a given system to a given time.
///
/// # Arguments
///
/// * `params` - System parameters.
/// * `n0` - Initial population size.
/// * `t` - Time to sample the population size at.
/// * `rng` - Random number generator to use.
///
/// # Examples
///
/// ```
/// use popfeedback::{Parameters, sample_branching_at_time};
/// use rand::rng;
/// let par = Parameters {a1: 1., a2: 1., b1: 0., b2: 0., I: 500., multiplicity: vec![0.5, 0.5]};
/// sample_branching_at_time(&par, 1000, 10., &mut rng());
///
/// ```
///
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

/// High order function to take sampling function that does one sample at a time and use it to
/// sample at multiple consequtive times, starting at a given initial population.
/// The system parameters are expected to be baked into the sampling function.
///
/// # Arguments
///
/// * n0 - Initial population size. Usually a u32 or the like.
/// * times - A slice of times at which we want to sample the population in.
/// * f - The sampling function that can sample a new population at a future time.
///
/// # Examples
///
/// ```
/// use popfeedback::{Parameters, sample_sde_at_time, sample_at_times};
/// use rand::rng;
/// let par = Parameters {a1: 1., a2: 1., b1: 0., b2: 0., I: 500., multiplicity: vec![0.5, 0.5]};
/// let func = |x: f64, y| sample_sde_at_time(&par, x, y, 1e-4, &mut rng());
/// sample_at_times(1000., &vec![1., 2., 3., 4., 5., 6., 7.], func);
///
/// ```
///
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

    #[test]
    fn zero_population_branching_no_immigration_is_zero_forever() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1.,
            b2: 1.,
            I: 0.,
            multiplicity: vec![1.],
        };
        let pop = super::sample_branching_at_time(&p, 0, 20., &mut rand::rng());
        assert_eq!(pop, 0);
    }

    #[test]
    fn zero_population_sde_no_immigration_is_zero_forever() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1.,
            b2: 1.,
            I: 0.,
            multiplicity: vec![1.],
        };
        let pop = super::sample_sde_at_time(&p, 0, 20., 1e-4, &mut rand::rng());
        assert_eq!(pop, 0.);
    }

    #[test]
    fn sample_size_matches_length_of_times_sde() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1e-8,
            b2: 1e-4,
            I: 400.,
            multiplicity: vec![0.25, 0.75],
        };
        let sde = |x: f64, y| super::sample_sde_at_time(&p, x, y, 1e-4, &mut rand::rng());
        let times = [10., 20., 30., 40., 50.];
        let pop = super::sample_at_times(5000_u32, &times, sde);
        assert_eq!(pop.len(), times.len());
    }

    #[test]
    fn sample_size_matches_length_of_times_branching() {
        let p = super::Parameters {
            a1: 1.,
            a2: 1.,
            b1: 1e-8,
            b2: 1e-4,
            I: 400.,
            multiplicity: vec![0.25, 0.75],
        };
        let branch = |x, y| super::sample_branching_at_time(&p, x, y, &mut rand::rng());
        let times = [10., 20., 30., 40., 50.];
        let pop = super::sample_at_times(5000_u32, &times, branch);
        assert_eq!(pop.len(), times.len());
    }
}
