use clap::{Args, Parser, Subcommand};
use popfeedback::Parameters;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Which model to use
    #[command(subcommand)]
    model: Model,

    /// Random number generator seed
    #[arg(short, long, num_args = 1.., value_delimiter = ' ', value_name = "Seed", default_values = ["48"])]
    seeds: Vec<u64>,

    /// Population size
    #[arg(short, long, value_name = "PopSize")]
    n: u32,

    /// Verbose flag
    #[arg(short, long)]
    v: bool,

    /// Debug flag
    #[arg(short, long)]
    debug: bool,
}

#[derive(Debug, Subcommand)]
enum Model {
    /// Branching process
    Branching {
        #[command(flatten)]
        ex: ExArgs,
    },

    /// Stochastic Differential Equation
    Sde {
        /// Time step size
        #[arg(value_name = "delt")]
        dt: f64,

        #[command(flatten)]
        ex: ExArgs,
    },
}

#[derive(Debug, Args)]
struct ExArgs {
    /// Sample times
    #[arg(short, long, num_args = 1.., value_delimiter = ' ', value_name = "Times")]
    times: Vec<f64>,

    #[command(flatten)]
    params: Params,
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

    let (dt, ex) = match &args.model {
        Model::Sde { dt, ex } => (dt, ex),
        Model::Branching { ex } => (&0., ex),
    };

    let par = &ex.params;
    let params = if let Some(path) = par.path.as_ref() {
        Parameters::from_json_file(path)
    } else {
        let v: Vec<f64> = par.values.as_deref().unwrap().to_vec();
        Parameters::from_vec(&v)
    };

    if args.v {
        println!("{:?}", &params);
        println!("Seeds: {:?}", seeds);
        println!("Initial population: {}", args.n);
        println!("Times: {:?}", &ex.times);
        match &args.model {
            Model::Sde { .. } => println!("Time step: {}", dt),
            Model::Branching { .. } => (),
        }
    }

    if args.debug {
        return;
    }

    for seed in &seeds {
        let mut rng = ChaCha20Rng::seed_from_u64(*seed);

        let pops = match &args.model {
            Model::Branching { .. } => {
                let fun = |x, y| popfeedback::sample_branching_at_time(&params, x, y, &mut rng);
                popfeedback::sample_at_times(args.n, &ex.times, fun)
            }
            Model::Sde { .. } => {
                let fun = |x: f64, y| popfeedback::sample_sde_at_time(&params, x, y, *dt, &mut rng);
                popfeedback::sample_at_times(args.n, &ex.times, fun)
            }
        };
        println!("{:?}", pops);
    }
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;
    Cli::command().debug_assert()
}
