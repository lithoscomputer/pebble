//! Applications and Pebble run the same public contract checks.

use std::path::PathBuf;
use std::{env, fs};

use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::test_support::{
    EnvironmentContract, EnvironmentContractError, MockEnvironment,
};
use uuid::Uuid;

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        Self(env::temp_dir().join(format!("pebble-environment-contract-{}", Uuid::new_v4())))
    }

    fn environment(&self) -> LocalEnvironment {
        LocalEnvironment::new(&self.0)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn local_files_satisfy_the_public_contract() {
    let scratch = Scratch::new();
    let env = scratch.environment();
    EnvironmentContract::new(&env, "fixtures")
        .verify_files()
        .await
        .expect("file contract");
}

#[tokio::test]
async fn local_search_satisfies_the_public_contract() {
    let scratch = Scratch::new();
    let env = scratch.environment();
    EnvironmentContract::new(&env, "fixtures")
        .verify_search()
        .await
        .expect("search contract");
}

#[tokio::test]
async fn local_commands_satisfy_the_public_contract() {
    let scratch = Scratch::new();
    let env = scratch.environment();
    EnvironmentContract::new(&env, "fixtures")
        .verify_commands()
        .await
        .expect("command contract");
}

#[tokio::test]
async fn a_static_mock_is_rejected_with_the_failed_check() {
    let env = MockEnvironment::linux();
    let error = EnvironmentContract::new(&env, "fixtures")
        .verify_files()
        .await
        .expect_err("a static double does not store writes");
    assert!(
        matches!(error, EnvironmentContractError::Operation {
            check: "read bytes",
            ..
        }),
        "{error:?}"
    );
}
