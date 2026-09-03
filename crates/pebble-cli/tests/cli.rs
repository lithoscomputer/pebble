//! The binary's behavior, as a person at a terminal sees it.
//!
//! Every case is a `trycmd` file under `tests/cmd/`: the arguments, what the
//! command prints on each stream, and how it exits. A case that runs a prompt
//! talks to a deterministic OpenAI-compatible server started in this process,
//! so the real codec and transport are exercised with no key and no network.
//!
//! The server binds an ephemeral port, and the harness hands its address to
//! every case through `PEBBLE_<PROVIDER>_BASE_URL` for each provider a case
//! may name, so two suites can run at once and a stale process can never take
//! the port. Each case names its own API key, which the server treats as a
//! namespace: the scenarios in `tests/cmd/scenarios.json` are seeded per
//! namespace, so one case cannot consume another's script.
//!
//! Cases run with a cleared environment, so a key in the developer's shell
//! cannot reach a real provider. `PATH` is passed through: the agent's tools
//! run under the Bash it finds there.

use std::env;
use std::path::PathBuf;

use tokio::net::TcpListener;
use twin_openai::config::Config;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_cases() {
    let mut config = Config::from_lookup(&|_| None).expect("the default twin configuration builds");
    config.scenarios_path =
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cmd/scenarios.json"));
    // A developer debugging a case can ask the twin for every request it saw.
    if let Some(path) = env::var_os("PEBBLE_TWIN_REQUEST_LOG") {
        config.request_log_path = Some(PathBuf::from(path));
    }
    let app = twin_openai::build_app_with_config(config).expect("the twin app builds");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("an ephemeral port is available");
    let address = listener.local_addr().expect("the listener has an address");
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("the twin serves");
    });

    let cases = trycmd::TestCases::new();
    let base_url = format!("http://{address}/v1");
    for provider in ["OPENAI", "MOONSHOT", "OPENROUTER", "VENICE", "FIREWORKS"] {
        cases.env(format!("PEBBLE_{provider}_BASE_URL"), base_url.clone());
    }
    cases
        .env("PATH", env::var("PATH").unwrap_or_default())
        .case("tests/cmd/*.toml");
    // Under coverage instrumentation the binary writes its profile where this
    // variable says; without it a case's cleared environment would drop the
    // binary's counts on the floor.
    if let Ok(profile_file) = env::var("LLVM_PROFILE_FILE") {
        cases.env("LLVM_PROFILE_FILE", profile_file);
    }
    cases.run();

    server.abort();
}
