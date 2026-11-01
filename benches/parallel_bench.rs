//! Compares a serial loop over seeds to `sample_seeds` with growing numbers of threads, for both
//! the SDE and the branching process, and checks that all of them give the very same sample paths.
//!
//! Run it on an otherwise idle machine with `cargo bench --bench parallel`.
//!
//! Exits with an error if the paths differ, or if using all of the CPUs isn't faster.

use popfeedback::Parameters;
use popfeedback::{sample_at_times, sample_branching_at_time, sample_sde_at_time, sample_seeds};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

const REPEATS: usize = 5;

fn median_time<T>(mut f: impl FnMut() -> T) -> Duration {
    let mut times: Vec<Duration> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            black_box(f());
            start.elapsed()
        })
        .collect();
    times.sort();
    times[REPEATS / 2]
}

fn compare<F>(name: &str, seeds: &[u64], path: F)
where
    F: Fn(&mut ChaCha20Rng) -> Vec<f64> + Sync,
{
    let serial = || -> Vec<Vec<f64>> {
        seeds
            .iter()
            .map(|seed| path(&mut ChaCha20Rng::seed_from_u64(*seed)))
            .collect()
    };
    let expected = serial();
    let serial_time = median_time(serial);

    println!("{name}: {} paths, median of {REPEATS} runs", seeds.len());
    println!("{:>10} {:>10} {:>8}", "threads", "time [s]", "speedup");
    println!(
        "{:>10} {:>10.3} {:>8}",
        "serial",
        serial_time.as_secs_f64(),
        "-"
    );

    let max_threads = available_parallelism().unwrap().get();
    let mut counts: Vec<usize> = (0..)
        .map(|x| 1 << x)
        .take_while(|x| *x < max_threads)
        .collect();
    counts.push(max_threads);

    let mut last_time = serial_time;
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let parallel = || pool.install(|| sample_seeds(seeds, &path));
        assert_eq!(
            expected,
            parallel(),
            "{name}: paths changed on {threads} threads"
        );
        last_time = median_time(parallel);
        println!(
            "{:>10} {:>10.3} {:>7.2}x",
            threads,
            last_time.as_secs_f64(),
            serial_time.as_secs_f64() / last_time.as_secs_f64()
        );
    }
    println!();
    assert!(
        max_threads == 1 || last_time < serial_time,
        "{name}: {max_threads} threads were not faster than a serial loop"
    );
}

fn main() {
    let p = Parameters {
        a1: 1.,
        a2: 1.,
        b1: 0.,
        b2: 1e-4,
        I: 500.,
        multiplicity: vec![0.5, 0.5],
    };
    let times = [2., 4., 6., 8., 10.];
    let seeds: Vec<u64> = (0..64).collect();

    compare("SDE", &seeds, |rng: &mut ChaCha20Rng| {
        let sde = |x: f64, y| sample_sde_at_time(&p, x, y, 1e-5, rng);
        sample_at_times(5000_u32, &times, sde)
    });
    compare("Branching", &seeds, |rng: &mut ChaCha20Rng| {
        let branch = |x, y| sample_branching_at_time(&p, x, y, rng);
        sample_at_times(5000_u32, &times, branch)
    });
}
