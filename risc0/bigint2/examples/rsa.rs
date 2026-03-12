// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use clap::Parser;
use num_bigint::BigUint;
use risc0_bigint2_methods::RSA_ELF;
use risc0_zkvm::{default_prover, ExecutorEnv};

#[derive(Parser)]
struct Args {
    /// Enable puffin
    #[arg(long, short)]
    puffin: bool,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    if args.puffin {
        puffin::set_scopes_on(true);
        let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
        println!("Puffin server: {server_addr}");
        let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();

        println!("Press enter to start");
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        println!("proving rsa...");
    }

    // base^65537 mod modulus (simple test case: 1^65537 mod N = 1 for any N)
    let base = BigUint::from(1u32);
    let modulus = BigUint::parse_bytes(
        b"a79633c36a2e9a91cac9505c300edb9a0aa8105f115856024fe27dac4910ab2f\
          ad99fe3700ddf532a3c86c5171d28177d55b3112fe73f46917a58006ac747ae3\
          96557fe84e62ba0cefd55858258e2aecd756a855ca7bd81be968fc3b908a3105\
          08828275065ab91b6f8f7964ef5684ee7fd5188c097a99323dfbe1302ab8723d\
          97f9d1a30b89b9fb7504e8fe1e85f858ae2f07818ff8c89f7f9dbd051e52f5da\
          e0eb33ab85d826906817f6712003c866f878874c503ec4ee02457a3c7012c6c7\
          7f66bfe23518d2eddbbe9fc3dbb8ad944d9a38e333c7caa86138f5121b895630\
          67c4bac9c32a26cd7c21830315f8e1c5b3783104f24648f7bf8556500b04d8bf\
          11bc7e777aa5e0653c5af9b085c2c08a16f7d2b4bc323d1b32a5cc8f4ab39f37\
          4d30efa01f8009aeb60358e5f22345fea8a123e46f394bcc669d183348b6eb29\
          fa5f972ebe90aa64684973f4e67c0ea7d82c5a05c4a837a0a63c86b201c54cbc\
          b1dbabf26a5bbd533b0ec91892716b7deb54c061d2e8696391f210e16b4714dd",
        16,
    )
    .unwrap();

    let env = ExecutorEnv::builder()
        .write(&(&base, &modulus))
        .unwrap()
        .build()
        .unwrap();
    let prover = default_prover();
    let receipt = prover.prove(env, RSA_ELF).unwrap().receipt;
    let result: BigUint = receipt.journal.decode().unwrap();
    println!("RSA modpow_65537({base}, {modulus}) = {result}");

    if args.puffin {
        puffin::GlobalProfiler::lock().new_frame();
    }
}
