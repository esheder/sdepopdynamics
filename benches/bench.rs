#![feature(test)]

extern crate test;

use popfeedback::{sample_branching_at_time, sample_sde_at_time, Parameters};
use rand::rng;
use test::{black_box, Bencher};

#[bench]
fn bench_sde_lowpop(b: &mut Bencher) {
    let p = Parameters {
        a1: 10.,
        a2: 30.,
        b1: 0.,
        b2: 0.,
        I: 3000.,
        multiplicity: vec![0.5, 0.5],
    };
    let mut rng = rng();
    b.iter(|| black_box(sample_sde_at_time(&p, 4000, 20., 1e-4, &mut rng)));
}

#[bench]
fn bench_branching_lowpop(b: &mut Bencher) {
    let p = Parameters {
        a1: 10.,
        a2: 30.,
        b1: 0.,
        b2: 0.,
        I: 3000.,
        multiplicity: vec![0.5, 0.5],
    };
    let mut rng = rng();
    b.iter(|| black_box(sample_branching_at_time(&p, 4000, 20., &mut rng)));
}

#[bench]
fn bench_sde_highpop(b: &mut Bencher) {
    let p = Parameters {
        a1: 60.,
        a2: 30.,
        b1: 0.01,
        b2: 0.,
        I: 5.,
        multiplicity: vec![0.5, 0.5],
    };
    let mut rng = rng();
    b.iter(|| black_box(sample_sde_at_time(&p, 4000, 20., 1e-4, &mut rng)));
}

#[bench]
fn bench_branching_highpop(b: &mut Bencher) {
    let p = Parameters {
        a1: 60.,
        a2: 30.,
        b1: 0.01,
        b2: 0.,
        I: 5.,
        multiplicity: vec![0.5, 0.5],
    };
    let mut rng = rng();
    b.iter(|| black_box(sample_branching_at_time(&p, 4000, 20., &mut rng)));
}
