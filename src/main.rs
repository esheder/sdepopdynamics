use clap::{Args, Parser, Subcommand};
use indicatif::ProgressBar;
use polars::prelude::{df, DataFrame, ParquetWriter};
use popfeedback::Parameters;
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Which case to run
    #[command(subcommand)]
    simulation: SimType,

    /// Random number generator seed
    #[arg(short, long, num_args = 1.., value_delimiter = ' ', value_name = "Seed", default_values = ["48"])]
    seeds: Vec<u64>,

    /// Population size
    #[arg(short, long, value_name = "PopSize")]
    n: u32,

    /// Verbose flag
    #[arg(short, long)]
    v: bool,

    /// Output file path
    #[arg(short, long, value_name = "Output file")]
    o: Option<std::path::PathBuf>,

    /// Debug flag
    #[arg(short, long)]
    debug: bool,

    /// Number of threads to sample paths on, results do not depend on it [default: all CPUs]
    #[arg(short = 'j', long, value_name = "Threads")]
    threads: Option<usize>,

    /// Model parameters
    #[command(flatten)]
    params: Params,
}

#[derive(Debug, Subcommand)]
enum SimType {
    /// Dynamic sampling
    DynSample {
        /// Sampling model
        #[command(subcommand)]
        model: Model,

        /// Sample times
        #[arg(short, long, num_args = 1.., value_delimiter = ' ', value_name = "Times")]
        times: Vec<f64>,
    },

    /// Exit time
    ExitTime {
        /// Sampling model
        #[command(subcommand)]
        model: Model,

        /// Low population bound
        #[arg(long, value_name = "LowerBound")]
        low: u32,

        /// High population bound
        #[arg(long, value_name = "UpperBound")]
        high: u32,

        /// Time step size
        #[arg(long, value_name = "TimeStep")]
        step: f64,
    },
}

#[derive(Debug, Subcommand)]
enum Model {
    /// Branching process
    Branching {},

    /// Stochastic Differential Equation
    Sde {
        /// Time step size
        #[arg(value_name = "delt")]
        dt: f64,
    },
}

#[derive(Args, Debug)]
#[group(required = true, multiple = false)]
struct Params {
    /// Path to JSON file with parameter setup
    #[arg(long, value_name = "Path")]
    path: Option<std::path::PathBuf>,

    /// Direct parameter setup with floats
    #[arg(long, num_args = 6.., value_delimiter = ' ', value_name = "f")]
    values: Option<Vec<f64>>,
}

fn main() {
    let args = Cli::parse();
    let seeds = match args.seeds.len() {
        2 => (args.seeds[0]..args.seeds[1]).collect(),
        _ => args.seeds,
    };

    let params = if let Some(path) = &args.params.path.as_ref() {
        Parameters::from_json_file(path)
    } else {
        let v: Vec<f64> = args.params.values.as_deref().unwrap().to_vec();
        Parameters::from_vec(&v)
    };

    if args.v {
        println!("{:?}", params);
        if args.o.is_some() {
            println!("Output file: {:?}", args.o.clone().unwrap())
        }
        println!("Seeds: {seeds:?}");
        println!("Initial population: {}", args.n);
        match args.simulation {
            SimType::DynSample {
                ref model,
                ref times,
            } => {
                println!("Running dynamics at a given vector of times.");
                println!("Times: {:?}", times);
                match model {
                    Model::Sde { dt } => {
                        println!("Using SDE model with an internal time step.");
                        println!("Time step: {dt:?}");
                    }
                    Model::Branching { .. } => println!("Using a branching process."),
                }
            }
            SimType::ExitTime {
                ref model,
                low,
                high,
                step,
            } => {
                println!("Finding exit time from a given bound.");
                println!("Low bound: {}", low);
                println!("Upper bound: {}", high);
                println!("Time step size: {}", step);
                match model {
                    Model::Sde { dt } => {
                        println!("Using SDE model with an internal time step.");
                        println!("Time step: {dt:?}");
                    }
                    Model::Branching { .. } => println!("Using a branching process."),
                }
            }
        }
    }

    if args.debug {
        return;
    }

    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("The global thread pool is only set up here");
    }

    let progress = ProgressBar::new(seeds.len() as u64);

    let outfile = args
        .o
        .map(|p| std::fs::File::create(p).expect("Output path should be openable"));

    let mut df: DataFrame = match args.simulation {
        SimType::ExitTime {
            model,
            low,
            high,
            step,
        } => {
            let (results, computation_times): (Vec<f64>, Vec<f64>) =
                popfeedback::sample_seeds(&seeds, |rng: &mut ChaCha20Rng| {
                    let start = Instant::now();
                    let exit_time = match model {
                        Model::Branching { .. } => {
                            let fun =
                                |x, y| popfeedback::sample_branching_at_time(&params, x, y, rng);
                            popfeedback::exit_time(args.n, low, high, step, fun)
                        }
                        Model::Sde { dt } => {
                            let fun =
                                |x, y| popfeedback::sample_sde_at_time(&params, x, y, dt, rng);
                            let n0: f64 = args.n.into();
                            popfeedback::exit_time(n0, low, high, step, fun)
                        }
                    };
                    progress.inc(1);
                    (exit_time, start.elapsed().as_secs_f64())
                })
                .into_iter()
                .unzip();
            df!(
            "Seed" => seeds,
            "Exit Time" => results,
            "Computation Time" => computation_times,
                    )
            .expect("Data is created by us and should never fail")
        }
        SimType::DynSample { model, times } => {
            let results: Vec<Vec<f64>> =
                popfeedback::sample_seeds(&seeds, |rng: &mut ChaCha20Rng| {
                    let path = match model {
                        Model::Branching { .. } => {
                            let fun =
                                |x, y| popfeedback::sample_branching_at_time(&params, x, y, rng);
                            popfeedback::sample_at_times(args.n, &times, fun)
                        }
                        Model::Sde { dt } => {
                            let fun =
                                |x: f64, y| popfeedback::sample_sde_at_time(&params, x, y, dt, rng);
                            popfeedback::sample_at_times(args.n, &times, fun)
                        }
                    };
                    progress.inc(1);
                    path
                });
            df!(
        "Seed" => seeds.iter().zip(std::iter::repeat(times.len())).flat_map(|(v, n)| std::iter::repeat_n(v,n)).copied().collect::<Vec<u64>>(),
        "Time" => times.iter().cycle().take(times.len()*seeds.len()).copied().collect::<Vec<f64>>(),
        "Population" => results.into_iter().flatten().collect::<Vec<f64>>()).expect("Data is created by us and should never fail")
        }
    };
    progress.finish();
    if let Some(mut f) = outfile {
        let _ = ParquetWriter::new(&mut f)
            .finish(&mut df)
            .expect("We gave it a valid file");
    } else {
        println!("{df:?}");
    }
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;
    Cli::command().debug_assert()
}
