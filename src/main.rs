use clap::{Args, Parser, Subcommand};
use indicatif::ProgressIterator;
use polars::prelude::{df, DataFrame, ParquetWriter};
use popfeedback::Parameters;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

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
        #[arg(short, long, value_name = "LowerBound")]
        low: u32,

        /// High population bound
        #[arg(short, long, value_name = "UpperBound")]
        high: u32,
    },
}

#[derive(Debug, Subcommand)]
enum Model {
    /// Branching process
    Branching {
        #[command(flatten)]
        ex: Params,
    },

    /// Stochastic Differential Equation
    Sde {
        /// Time step size
        #[arg(value_name = "delt")]
        dt: f64,

        #[command(flatten)]
        ex: Params,
    },
}

fn par_from_model(model: Model) -> Parameters {
    let ex = match model {
        Model::Sde { dt: _, ex } => ex,
        Model::Branching { ex } => ex,
    };
    if let Some(path) = &ex.path.as_ref() {
        Parameters::from_json_file(path)
    } else {
        let v: Vec<f64> = ex.values.as_deref().unwrap().to_vec();
        Parameters::from_vec(&v)
    }
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

    let model = match args.simulation {
        SimType::DynSample { model, times: _ } => model,
        SimType::ExitTime {
            model,
            low: _,
            high: _,
        } => model,
    };

    let (dt, params) = match model {
        Model::Sde { dt, ex } => (dt, ex),
        Model::Branching { ex } => (0., ex),
    };

    if args.v {
        println!("{:?}", params);
        if args.o.is_some() {
            println!("Output file: {:?}", args.o.clone().unwrap())
        }
        println!("Seeds: {seeds:?}");
        println!("Initial population: {}", args.n);
        match args.simulation {
            SimType::DynSample { model: _, times } => {
                println!("Running dynamics at a given vector of times.");
                println!("Times: {:?}", times);
            }
            SimType::ExitTime {
                model: _,
                low,
                high,
            } => {
                println!("Finding exit time from a given bound.");
                println!("Low bound: {}", low);
                println!("Upper bound: {}", high);
            }
        }
        match model {
            Model::Sde { .. } => {
                println!("Using SDE model with an internal time step.");
                println!("Time step: {dt:?}");
            }
            Model::Branching { .. } => println!("Using a branching process."),
        }
    }

    if args.debug {
        return;
    }

    let outfile = args
        .o
        .map(|p| std::fs::File::create(p).expect("Output path should be openable"));

    let params = par_from_model(model);

    let mut df: DataFrame = match args.simulation {
        SimType::ExitTime { model, low, high } => {
            let results: Vec<f64> = seeds
                .iter()
                .progress()
                .map(|seed| {
                    let mut rng = ChaCha20Rng::seed_from_u64(*seed);
                    let f = match model {
                        Model::Branching { .. } => {
                            |x, y| popfeedback::sample_branching_at_time(&params, x, y, &mut rng)
                        }
                        Model::Sde { .. } => {
                            |x, y| popfeedback::sample_sde_at_time(&params, x, y, dt, &mut rng)
                        }
                    };
                    popfeedback::exit_time(args.n, low, high, dt, f)
                })
                .collect();
            df!()
        }
        SimType::DynSample { model, times } => {
            let results: Vec<Vec<f64>> = seeds
                .iter()
                .progress()
                .map(|seed| {
                    let mut rng = ChaCha20Rng::seed_from_u64(*seed);

                    match model {
                        Model::Branching { .. } => {
                            let fun = |x, y| {
                                popfeedback::sample_branching_at_time(&params, x, y, &mut rng)
                            };
                            popfeedback::sample_at_times(args.n, &times, fun)
                        }
                        Model::Sde { .. } => {
                            let fun = |x: f64, y| {
                                popfeedback::sample_sde_at_time(&params, x, y, dt, &mut rng)
                            };
                            popfeedback::sample_at_times(args.n, &times, fun)
                        }
                    }
                })
                .collect();
            df!(
        "Seed" => seeds.iter().zip(std::iter::repeat(times.len())).flat_map(|(v, n)| std::iter::repeat_n(v,n)).copied().collect::<Vec<u64>>(),
        "Time" => times.iter().cycle().take(times.len()*seeds.len()).copied().collect::<Vec<f64>>(),
        "Population" => results.into_iter().flatten().collect::<Vec<f64>>()).expect("Data is created by us and should never fail")
        }
    };
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
