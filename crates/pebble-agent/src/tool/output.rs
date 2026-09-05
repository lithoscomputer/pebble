//! Model content and application-owned information from one tool call.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An artifact stored by the embedding application.
///
/// `reference` is opaque. It need not be a local path or a URL. The application
/// owns authorization, retrieval, retention, and the meaning of the reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolArtifact {
    /// Application-defined retrieval reference.
    pub reference:   String,
    /// Short description of the artifact.
    pub label:       String,
    /// MIME type of the stored bytes.
    pub media_type:  String,
    /// Number of stored bytes.
    pub byte_length: u64,
}

/// Tool information for middleware and observers, excluded from model requests.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolOutputMetadata {
    /// Application-defined structured details. The application owns redaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details:   Option<Value>,
    /// Artifacts produced by the call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ToolArtifact>,
}

impl ToolOutputMetadata {
    /// Whether this call has no observer-only information.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.details.is_none() && self.artifacts.is_empty()
    }
}
