//! Paths as the adapter resolves them inside a sandbox: a relative path is
//! against the session's working directory, which may sit below the
//! provider's own.

use std::path::Path;

/// `path` as the driver will see it: absolute as given, relative against
/// `working_dir`.
#[must_use]
pub fn resolve_path(path: &str, working_dir: &str) -> String {
    if Path::new(path).is_absolute() {
        path.to_string()
    } else {
        join_sandbox_path(working_dir, path)
    }
}

/// `relative_path` under `base` with one separator between them; either
/// side empty yields the other.
#[must_use]
pub fn join_sandbox_path(base: &str, relative_path: &str) -> String {
    if relative_path.is_empty() {
        return base.to_string();
    }
    if base.is_empty() {
        return relative_path.to_string();
    }
    if base == "/" {
        return format!("/{relative_path}");
    }
    format!("{}/{relative_path}", base.trim_end_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_resolve_against_the_working_directory() {
        assert_eq!(resolve_path("src/main.rs", "/work"), "/work/src/main.rs");
        assert_eq!(resolve_path("/etc/hosts", "/work"), "/etc/hosts");
        assert_eq!(resolve_path("", "/work"), "/work");
    }

    #[test]
    fn joins_keep_one_separator() {
        assert_eq!(join_sandbox_path("/work/", "a"), "/work/a");
        assert_eq!(join_sandbox_path("/", "a"), "/a");
        assert_eq!(join_sandbox_path("", "a"), "a");
        assert_eq!(join_sandbox_path("/work", ""), "/work");
    }
}
