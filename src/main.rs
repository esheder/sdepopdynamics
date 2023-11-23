use clap::{Parser, Subcommand};
use popfeedback::{Parameters};
use popfeedback;
use std::path::PathBuf;
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Parameters Configuration File
    #[arg(value_name = "ParamFile")]
    pfile: PathBuf,

    /// Population size
    #[arg(value_name = "PopSize")]
    n0: u32,

    /// Random number generator seed
    #[arg(short, long, value_name = "Seed", default_value="48")]
    seed: u64,

    /// Which model to use
    #[command(subcommand)]
    model: Model,
}

#[derive(Debug, Subcommand)]
enum Model {
    /// Branching process
    Branching {
        /// Times at which to sample the result
        #[arg(value_name = "Times")]
        times: Vec<f64>,
    },

    /// Stochastic Differential Equation
    SDE {
        /// Time step size
        #[arg(value_name = "delt")]
        dt: f64,

        /// Times at which to sample the result
        #[arg(value_name = "Times")]
        times: Vec<f64>,
    },
}


fn main() {
    let args = Cli::parse();
    let params = Parameters::from_json_file(&args.pfile);
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    
    let pops = match &args.model {
        Model::Branching{times} => {
            let fun = |x, y| popfeedback::sample_branching_at_time(&params, x, y, &mut rng);
            popfeedback::sample_at_times(args.n0, &times, fun)
        }
        Model::SDE{dt, times} => {
            let fun = |x: f64, y| popfeedback::sample_sde_at_time(&params, x, y, *dt, &mut rng);
            popfeedback::sample_at_times(args.n0, &times, fun)
        },
    };
    
    println!("{:?}", &args.model);
    println!("{:?}", pops);

}
