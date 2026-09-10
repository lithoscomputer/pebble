//! Shared configuration and credential behavior through the real CLI.

use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::{env, fs};

use serde_json::json;
use tokio::io::AsyncWriteExt as _;
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::task::JoinHandle;
use twin_openai::config::Config;

const KEY: &str = "exec-answers-without-tools";

#[tokio::test]
async fn exec_applies_project_defaults_and_explicit_model_precedence() {
    let home = tempfile::tempdir().unwrap();
    let repo = home.path().join("repo");
    let nested = repo.join("src");
    fs::create_dir_all(&nested).unwrap();
    fs::create_dir(repo.join(".git")).unwrap();
    fs::create_dir(repo.join(".pebble")).unwrap();
    fs::write(
        home.path().join("settings.json"),
        r#"{"model":"invalid-global"}"#,
    )
    .unwrap();
    let path = repo.join(".pebble/settings.json");
    fs::write(&path, r#"{"model":"gpt-5.6","reasoning":"high"}"#).unwrap();
    let (url, server) = provider().await;
    let output = command(home.path(), &url)
        .env("OPENAI_API_KEY", KEY)
        .args(["exec", "say hello", "--quiet", "--cwd"])
        .arg(&nested)
        .output()
        .await
        .unwrap();
    assert_eq!(success(output), "Hello from the twin.\n");
    server.abort();
    let _ = server.await;
    let (url, server) = provider().await;
    fs::write(&path, r#"{"model":"invalid-project","reasoning":null}"#).unwrap();
    let output = command(home.path(), &url)
        .env("OPENAI_API_KEY", KEY)
        .args([
            "exec",
            "say hello",
            "--quiet",
            "--model",
            "gpt-5.6",
            "--cwd",
        ])
        .arg(&nested)
        .output()
        .await
        .unwrap();
    assert_eq!(success(output), "Hello from the twin.\n");
    server.abort();
    let _ = server.await;
}

fn command(home: &Path, url: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pebble"));
    command
        .env_clear()
        .env("PATH", env::var_os("PATH").unwrap_or_default())
        .env("PEBBLE_HOME", home)
        .env("PEBBLE_OPENAI_BASE_URL", url)
        .current_dir(home)
        .stdin(Stdio::null())
        .kill_on_drop(true);
    if let Some(profile) = env::var_os("LLVM_PROFILE_FILE") {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    command
}

fn success(output: Output) -> String {
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 command output");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 command errors");
    assert!(
        output.status.success(),
        "command failed: {stdout}\n{stderr}"
    );
    assert!(!stdout.contains(KEY));
    assert!(!stderr.contains(KEY));
    stdout
}

async fn login(home: &Path, provider: &str, key: &str) -> Output {
    let mut child = command(home, "http://127.0.0.1:1/v1")
        .args(["auth", "login", provider, "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start auth command");
    child
        .stdin
        .take()
        .expect("piped input")
        .write_all(key.as_bytes())
        .await
        .expect("write fixture key");
    child.wait_with_output().await.expect("auth command exits")
}

async fn provider() -> (String, JoinHandle<()>) {
    let mut config = Config::from_lookup(&|_| None).expect("twin config");
    config.scenarios_path =
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cmd/scenarios.json"));
    let app = twin_openai::build_app_with_config(config).expect("twin app");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture server");
    let url = format!(
        "http://{}/v1",
        listener.local_addr().expect("fixture address")
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve fixture");
    });
    (url, server)
}

fn write_auth(home: &Path, providers: &serde_json::Value) {
    fs::write(
        home.join("auth.json"),
        serde_json::to_vec(&json!({"version":1,"providers":providers})).expect("fixture JSON"),
    )
    .expect("write fixture credentials");
}

#[tokio::test]
async fn exec_uses_saved_settings_custom_catalog_and_saved_credentials() {
    let home = tempfile::tempdir().unwrap();
    fs::write(
        home.path().join("settings.json"),
        r#"{"model":"openai/custom"}"#,
    )
    .unwrap();
    fs::write(
        home.path().join("models.toml"),
        r#"
schema_version = 1
[providers.openai]
base_url = "http://127.0.0.1:1/v1"
[providers.openai.models.custom]
display_name = "Custom fixture"
api_model = "gpt-5.6"
capabilities = { text = true, tools = true }
metadata.agent.profile = "gpt56"
"#,
    )
    .unwrap();
    success(login(home.path(), "openai", KEY).await);
    let (url, server) = provider().await;
    // The environment endpoint wins over the file endpoint. The saved model
    // resolves through the custom catalog row to a real provider request.
    let output = command(home.path(), &url)
        .args(["exec", "say hello", "--quiet"])
        .output()
        .await
        .unwrap();
    assert_eq!(success(output), "Hello from the twin.\n");
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn conventional_environment_credentials_override_a_saved_key() {
    let home = tempfile::tempdir().unwrap();
    // An explicit model also wins over an unusable saved default.
    fs::write(
        home.path().join("settings.json"),
        r#"{"model":"not-a-model"}"#,
    )
    .unwrap();
    success(login(home.path(), "openai", "unused-saved-account").await);
    let (url, server) = provider().await;
    let output = command(home.path(), &url)
        .env("OPENAI_API_KEY", KEY)
        .args(["exec", "say hello", "--model", "gpt-5.6", "--quiet"])
        .output()
        .await
        .unwrap();
    assert_eq!(success(output), "Hello from the twin.\n");
    let output = command(home.path(), &url)
        .env("OPENAI_API_KEY", KEY)
        .args(["auth", "status", "openai"])
        .output()
        .await
        .unwrap();
    assert!(success(output).contains("openai: environment"));
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn explicit_environment_sources_override_conventional_variables() {
    let home = tempfile::tempdir().unwrap();
    write_auth(
        home.path(),
        &json!({"openai":{"type":"env","variable":"PEBBLE_TEST_ACCOUNT"}}),
    );
    let (url, server) = provider().await;
    let output = command(home.path(), &url)
        .env("OPENAI_API_KEY", "wrong-account")
        .env("PEBBLE_TEST_ACCOUNT", KEY)
        .args(["exec", "say hello", "--model", "gpt-5.6", "--quiet"])
        .output()
        .await
        .unwrap();
    assert_eq!(success(output), "Hello from the twin.\n");
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn a_missing_explicit_source_does_not_fall_back_or_start_an_agent() {
    let home = tempfile::tempdir().unwrap();
    write_auth(
        home.path(),
        &json!({"openai":{"type":"env","variable":"PEBBLE_TEST_MISSING_KEY"}}),
    );
    let cwd = home.path().join("must-not-start");
    let output = command(home.path(), "http://127.0.0.1:1/v1")
        .env("OPENAI_API_KEY", KEY)
        .args(["exec", "say hello", "--model", "gpt-5.6", "--cwd"])
        .arg(&cwd)
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("PEBBLE_TEST_MISSING_KEY"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(KEY));
    assert!(!cwd.exists());
}

#[tokio::test]
async fn logout_reports_remaining_environment_credentials() {
    let home = tempfile::tempdir().unwrap();
    success(login(home.path(), "openai", KEY).await);
    let output = command(home.path(), "http://127.0.0.1:1/v1")
        .env("OPENAI_API_KEY", KEY)
        .args(["auth", "logout", "openai"])
        .output()
        .await
        .unwrap();
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Still configured through environment")
    );
    success(output);
    let auth: serde_json::Value =
        serde_json::from_slice(&fs::read(home.path().join("auth.json")).unwrap()).unwrap();
    assert!(auth["providers"].as_object().unwrap().is_empty());
}

#[tokio::test]
async fn explicit_header_credentials_reach_the_provider() {
    let home = tempfile::tempdir().unwrap();
    fs::write(
        home.path().join("models.toml"),
        "schema_version = 1\n[providers.openai.auth]\ntype = 'headers'\n",
    )
    .unwrap();
    write_auth(
        home.path(),
        &json!({"openai":{"type":"headers","headers":{
            "authorization":{"type":"literal","value":format!("Bearer {KEY}")},
            "x-project":{"type":"env","variable":"PEBBLE_TEST_PROJECT"}
        }}}),
    );
    let (url, server) = provider().await;
    let output = command(home.path(), &url)
        .env("PEBBLE_TEST_PROJECT", "fixture-project")
        .args(["exec", "say hello", "--model", "gpt-5.6", "--quiet"])
        .output()
        .await
        .unwrap();
    assert_eq!(success(output), "Hello from the twin.\n");
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn concurrent_login_processes_do_not_erase_credentials() {
    let home = tempfile::tempdir().unwrap();
    let (first, second, third) = tokio::join!(
        login(home.path(), "openai", "fixture-openai"),
        login(home.path(), "anthropic", "fixture-anthropic"),
        login(home.path(), "gemini", "fixture-gemini")
    );
    success(first);
    success(second);
    success(third);
    let auth: serde_json::Value =
        serde_json::from_slice(&fs::read(home.path().join("auth.json")).unwrap()).unwrap();
    assert_eq!(auth["providers"].as_object().unwrap().len(), 3);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            fs::metadata(home.path().join("auth.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

#[tokio::test]
async fn invalid_credential_schema_never_echoes_the_offending_value() {
    let home = tempfile::tempdir().unwrap();
    write_auth(home.path(), &json!({"openai":{"type":KEY}}));
    let output = command(home.path(), "http://127.0.0.1:1/v1")
        .args(["auth", "status"])
        .output()
        .await
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("auth.json at line"));
    assert!(!stderr.contains(KEY));
}

#[tokio::test]
async fn stdin_accepts_a_full_size_key_with_a_final_newline_and_rejects_larger_input() {
    let home = tempfile::tempdir().unwrap();
    let key = "k".repeat(8192);
    success(login(home.path(), "openai", &format!("{key}\r\n")).await);
    let original = fs::read(home.path().join("auth.json")).unwrap();
    let output = login(home.path(), "openai", &format!("{key}x")).await;
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains(&key));
    assert_eq!(fs::read(home.path().join("auth.json")).unwrap(), original);
}
