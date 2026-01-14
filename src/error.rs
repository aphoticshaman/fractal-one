//! ═══════════════════════════════════════════════════════════════════════════════
//! ERROR — Unified Error Type for Fractal
//! ═══════════════════════════════════════════════════════════════════════════════
//! Centralized error handling. No scattered .unwrap() or .expect() calls.
//! ═══════════════════════════════════════════════════════════════════════════════

use std::fmt;

/// The unified error type for the Fractal crate
#[derive(Debug)]
pub enum FractalError {
    /// I/O error (file operations, network, etc.)
    Io(std::io::Error),
    /// JSON serialization/deserialization error
    Json(serde_json::Error),
    /// Authentication error
    Auth(AuthError),
    /// Authorization error
    Authorization(AuthorizationError),
    /// Containment/boundary violation
    Containment(ContainmentError),
    /// Deference required (action blocked pending approval)
    DeferenceRequired(DeferenceError),
    /// Configuration error
    Config(ConfigError),
    /// Validation error
    Validation(ValidationError),
    /// Internal error (should not happen)
    Internal(String),
}

impl std::error::Error for FractalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FractalError::Io(e) => Some(e),
            FractalError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for FractalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FractalError::Io(e) => write!(f, "I/O error: {}", e),
            FractalError::Json(e) => write!(f, "JSON error: {}", e),
            FractalError::Auth(e) => write!(f, "Authentication error: {}", e),
            FractalError::Authorization(e) => write!(f, "Authorization error: {}", e),
            FractalError::Containment(e) => write!(f, "Containment error: {}", e),
            FractalError::DeferenceRequired(e) => write!(f, "Deference required: {}", e),
            FractalError::Config(e) => write!(f, "Configuration error: {}", e),
            FractalError::Validation(e) => write!(f, "Validation error: {}", e),
            FractalError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl From<std::io::Error> for FractalError {
    fn from(err: std::io::Error) -> Self {
        FractalError::Io(err)
    }
}

impl From<serde_json::Error> for FractalError {
    fn from(err: serde_json::Error) -> Self {
        FractalError::Json(err)
    }
}

/// Authentication-specific errors
#[derive(Debug, Clone)]
pub enum AuthError {
    /// Credentials expired
    Expired,
    /// Invalid credentials
    InvalidCredentials,
    /// Missing credentials
    MissingCredentials,
    /// Credential type not accepted
    UnsupportedCredentialType(String),
    /// Identity not found
    IdentityNotFound(String),
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthError::Expired => write!(f, "Credentials have expired"),
            AuthError::InvalidCredentials => write!(f, "Invalid credentials"),
            AuthError::MissingCredentials => write!(f, "No credentials provided"),
            AuthError::UnsupportedCredentialType(t) => {
                write!(f, "Unsupported credential type: {}", t)
            }
            AuthError::IdentityNotFound(id) => write!(f, "Identity not found: {}", id),
        }
    }
}

impl std::error::Error for AuthError {}

impl From<AuthError> for FractalError {
    fn from(err: AuthError) -> Self {
        FractalError::Auth(err)
    }
}

/// Authorization-specific errors
#[derive(Debug, Clone)]
pub enum AuthorizationError {
    /// Insufficient authorization level
    InsufficientLevel {
        required: String,
        actual: String,
    },
    /// Operation not permitted
    OperationNotPermitted(String),
    /// Resource access denied
    AccessDenied(String),
}

impl fmt::Display for AuthorizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthorizationError::InsufficientLevel { required, actual } => {
                write!(
                    f,
                    "Insufficient authorization: required {}, have {}",
                    required, actual
                )
            }
            AuthorizationError::OperationNotPermitted(op) => {
                write!(f, "Operation not permitted: {}", op)
            }
            AuthorizationError::AccessDenied(resource) => {
                write!(f, "Access denied to resource: {}", resource)
            }
        }
    }
}

impl std::error::Error for AuthorizationError {}

impl From<AuthorizationError> for FractalError {
    fn from(err: AuthorizationError) -> Self {
        FractalError::Authorization(err)
    }
}

/// Containment-specific errors
#[derive(Debug, Clone)]
pub enum ContainmentError {
    /// Boundary violation
    BoundaryViolation {
        boundary: String,
        description: String,
    },
    /// Manipulation attempt detected
    ManipulationDetected {
        attempt_type: String,
        confidence: f64,
    },
    /// Threat level too high
    ThreatLevelExceeded(String),
}

impl fmt::Display for ContainmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainmentError::BoundaryViolation {
                boundary,
                description,
            } => {
                write!(f, "Boundary '{}' violated: {}", boundary, description)
            }
            ContainmentError::ManipulationDetected {
                attempt_type,
                confidence,
            } => {
                write!(
                    f,
                    "Manipulation attempt detected: {} (confidence: {:.2})",
                    attempt_type, confidence
                )
            }
            ContainmentError::ThreatLevelExceeded(level) => {
                write!(f, "Threat level exceeded: {}", level)
            }
        }
    }
}

impl std::error::Error for ContainmentError {}

impl From<ContainmentError> for FractalError {
    fn from(err: ContainmentError) -> Self {
        FractalError::Containment(err)
    }
}

/// Deference-specific errors
#[derive(Debug, Clone)]
pub struct DeferenceError {
    /// ID of the pending action
    pub action_id: u64,
    /// Description of the action awaiting approval
    pub action: String,
    /// Who needs to approve
    pub required_approver: String,
}

impl fmt::Display for DeferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Action '{}' (id: {}) requires approval from: {}",
            self.action, self.action_id, self.required_approver
        )
    }
}

impl std::error::Error for DeferenceError {}

impl From<DeferenceError> for FractalError {
    fn from(err: DeferenceError) -> Self {
        FractalError::DeferenceRequired(err)
    }
}

/// Configuration-specific errors
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Missing required field
    MissingField(String),
    /// Invalid value
    InvalidValue { field: String, message: String },
    /// File not found
    FileNotFound(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingField(field) => write!(f, "Missing required field: {}", field),
            ConfigError::InvalidValue { field, message } => {
                write!(f, "Invalid value for '{}': {}", field, message)
            }
            ConfigError::FileNotFound(path) => write!(f, "Config file not found: {}", path),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for FractalError {
    fn from(err: ConfigError) -> Self {
        FractalError::Config(err)
    }
}

/// Validation-specific errors
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Input validation failed
    InvalidInput { field: String, message: String },
    /// Constraint violation
    ConstraintViolation(String),
    /// Format error
    FormatError { expected: String, actual: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidInput { field, message } => {
                write!(f, "Invalid input for '{}': {}", field, message)
            }
            ValidationError::ConstraintViolation(msg) => write!(f, "Constraint violation: {}", msg),
            ValidationError::FormatError { expected, actual } => {
                write!(f, "Format error: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

impl From<ValidationError> for FractalError {
    fn from(err: ValidationError) -> Self {
        FractalError::Validation(err)
    }
}

/// Type alias for Result with FractalError
pub type FractalResult<T> = Result<T, FractalError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FractalError::Auth(AuthError::Expired);
        assert!(err.to_string().contains("expired"));

        let err = FractalError::Authorization(AuthorizationError::InsufficientLevel {
            required: "Administrator".to_string(),
            actual: "User".to_string(),
        });
        assert!(err.to_string().contains("Administrator"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let fractal_err: FractalError = io_err.into();
        assert!(matches!(fractal_err, FractalError::Io(_)));
    }
}
