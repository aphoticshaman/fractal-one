//! ═══════════════════════════════════════════════════════════════════════════════
//! AUTHENTICATION — Cryptographic Identity Verification
//! ═══════════════════════════════════════════════════════════════════════════════
//! Trust no string. Verify everything cryptographically.
//! An unauthenticated identity is no identity at all.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::time::TimePoint;
use std::collections::HashMap;

/// Cryptographic credential types
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CredentialType {
    /// No authentication (lowest trust)
    #[default]
    None,
    /// API key authentication
    ApiKey,
    /// Session token
    SessionToken,
    /// Certificate-based (X.509)
    Certificate,
    /// Multi-factor authentication
    MultiFactor,
    /// Hardware key (FIDO2/WebAuthn)
    HardwareKey,
}

/// Authorization level for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum AuthorizationLevel {
    /// Read-only access
    #[default]
    ReadOnly,
    /// Standard user operations
    User,
    /// Operator-level access
    Operator,
    /// Administrator access
    Administrator,
    /// Emergency override (requires hardware key + MFA)
    EmergencyOverride,
}

/// An authenticated identity with cryptographic verification
#[derive(Debug, Clone)]
pub struct AuthenticatedIdentity {
    /// Unique identifier (verified, not user-provided)
    pub id: String,
    /// How this identity was authenticated
    pub credential_type: CredentialType,
    /// Authorization level granted
    pub authorization_level: AuthorizationLevel,
    /// When authentication was verified
    pub verified_at: TimePoint,
    /// Authentication expiry time
    pub expires_at: Option<TimePoint>,
    /// Hash of the credential used (for audit, not the credential itself)
    pub credential_hash: String,
    /// Additional claims/attributes
    pub claims: HashMap<String, String>,
}

impl Default for AuthenticatedIdentity {
    fn default() -> Self {
        Self {
            id: String::new(),
            credential_type: CredentialType::None,
            authorization_level: AuthorizationLevel::ReadOnly,
            verified_at: TimePoint::now(),
            expires_at: None,
            credential_hash: String::new(),
            claims: HashMap::new(),
        }
    }
}

impl AuthenticatedIdentity {
    /// Create an unauthenticated identity (for anonymous access)
    pub fn anonymous() -> Self {
        Self {
            id: "anonymous".to_string(),
            credential_type: CredentialType::None,
            authorization_level: AuthorizationLevel::ReadOnly,
            verified_at: TimePoint::now(),
            expires_at: None,
            credential_hash: String::new(),
            claims: HashMap::new(),
        }
    }

    /// Check if identity is still valid
    pub fn is_valid(&self) -> bool {
        if let Some(expires) = &self.expires_at {
            TimePoint::now().duration_since(expires).as_secs() == 0
        } else {
            true
        }
    }

    /// Check if identity has at least the given authorization level
    pub fn has_authorization(&self, required: AuthorizationLevel) -> bool {
        self.is_valid() && self.authorization_level >= required
    }

    /// Check if identity can perform shutdown operations
    pub fn can_shutdown(&self) -> bool {
        self.has_authorization(AuthorizationLevel::Operator)
    }

    /// Check if identity can modify system values
    pub fn can_modify_values(&self) -> bool {
        self.has_authorization(AuthorizationLevel::Administrator)
    }

    /// Check if identity can use emergency override
    pub fn can_emergency_override(&self) -> bool {
        self.authorization_level == AuthorizationLevel::EmergencyOverride
            && self.credential_type >= CredentialType::MultiFactor
    }
}

/// Result of authentication attempt
#[derive(Debug, Clone)]
pub struct AuthenticationResult {
    pub success: bool,
    pub identity: Option<AuthenticatedIdentity>,
    pub error: Option<AuthenticationError>,
}

/// Authentication errors
#[derive(Debug, Clone)]
pub enum AuthenticationError {
    /// Invalid credentials
    InvalidCredentials,
    /// Credentials expired
    CredentialsExpired,
    /// Insufficient authorization
    InsufficientAuthorization,
    /// Identity not found
    IdentityNotFound,
    /// Rate limited
    RateLimited,
    /// Credential revoked
    CredentialRevoked,
}

impl std::fmt::Display for AuthenticationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCredentials => write!(f, "Invalid credentials"),
            Self::CredentialsExpired => write!(f, "Credentials expired"),
            Self::InsufficientAuthorization => write!(f, "Insufficient authorization"),
            Self::IdentityNotFound => write!(f, "Identity not found"),
            Self::RateLimited => write!(f, "Rate limited"),
            Self::CredentialRevoked => write!(f, "Credential revoked"),
        }
    }
}

/// Simple credential hasher (in production, use proper crypto)
pub fn hash_credential(credential: &str) -> String {
    // Simple hash for demonstration - in production use SHA-256 or better
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for byte in credential.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    format!("{:016x}", hash)
}

/// Authentication provider trait
pub trait AuthProvider: Send + Sync {
    /// Authenticate with API key
    fn authenticate_api_key(&self, key: &str) -> AuthenticationResult;

    /// Authenticate with session token
    fn authenticate_session(&self, token: &str) -> AuthenticationResult;

    /// Verify an existing identity is still valid
    fn verify_identity(&self, identity: &AuthenticatedIdentity) -> bool;

    /// Revoke credentials for an identity
    fn revoke(&mut self, identity_id: &str) -> bool;
}

/// In-memory authentication provider for testing/simple deployments
#[derive(Debug, Default)]
pub struct InMemoryAuthProvider {
    /// API key -> Identity mapping
    api_keys: HashMap<String, AuthenticatedIdentity>,
    /// Session token -> Identity mapping
    sessions: HashMap<String, AuthenticatedIdentity>,
    /// Revoked credential hashes
    revoked: Vec<String>,
}

impl InMemoryAuthProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an API key
    pub fn register_api_key(&mut self, key: &str, identity: AuthenticatedIdentity) {
        let hash = hash_credential(key);
        let mut identity = identity;
        identity.credential_hash = hash.clone();
        identity.credential_type = CredentialType::ApiKey;
        self.api_keys.insert(hash, identity);
    }

    /// Create a session for an identity
    pub fn create_session(&mut self, identity: AuthenticatedIdentity) -> String {
        let token = format!("session_{}", hash_credential(&format!("{:?}{:?}", identity.id, TimePoint::now())));
        let hash = hash_credential(&token);
        let mut session_identity = identity;
        session_identity.credential_hash = hash.clone();
        session_identity.credential_type = CredentialType::SessionToken;
        self.sessions.insert(hash, session_identity);
        token
    }
}

impl AuthProvider for InMemoryAuthProvider {
    fn authenticate_api_key(&self, key: &str) -> AuthenticationResult {
        let hash = hash_credential(key);

        if self.revoked.contains(&hash) {
            return AuthenticationResult {
                success: false,
                identity: None,
                error: Some(AuthenticationError::CredentialRevoked),
            };
        }

        match self.api_keys.get(&hash) {
            Some(identity) => {
                if identity.is_valid() {
                    AuthenticationResult {
                        success: true,
                        identity: Some(identity.clone()),
                        error: None,
                    }
                } else {
                    AuthenticationResult {
                        success: false,
                        identity: None,
                        error: Some(AuthenticationError::CredentialsExpired),
                    }
                }
            }
            None => AuthenticationResult {
                success: false,
                identity: None,
                error: Some(AuthenticationError::InvalidCredentials),
            },
        }
    }

    fn authenticate_session(&self, token: &str) -> AuthenticationResult {
        let hash = hash_credential(token);

        if self.revoked.contains(&hash) {
            return AuthenticationResult {
                success: false,
                identity: None,
                error: Some(AuthenticationError::CredentialRevoked),
            };
        }

        match self.sessions.get(&hash) {
            Some(identity) => {
                if identity.is_valid() {
                    AuthenticationResult {
                        success: true,
                        identity: Some(identity.clone()),
                        error: None,
                    }
                } else {
                    AuthenticationResult {
                        success: false,
                        identity: None,
                        error: Some(AuthenticationError::CredentialsExpired),
                    }
                }
            }
            None => AuthenticationResult {
                success: false,
                identity: None,
                error: Some(AuthenticationError::InvalidCredentials),
            },
        }
    }

    fn verify_identity(&self, identity: &AuthenticatedIdentity) -> bool {
        if self.revoked.contains(&identity.credential_hash) {
            return false;
        }
        identity.is_valid()
    }

    fn revoke(&mut self, identity_id: &str) -> bool {
        // Find and revoke all credentials for this identity
        let mut revoked_any = false;

        for (hash, identity) in &self.api_keys {
            if identity.id == identity_id {
                self.revoked.push(hash.clone());
                revoked_any = true;
            }
        }

        for (hash, identity) in &self.sessions {
            if identity.id == identity_id {
                self.revoked.push(hash.clone());
                revoked_any = true;
            }
        }

        revoked_any
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anonymous_identity() {
        let anon = AuthenticatedIdentity::anonymous();
        assert_eq!(anon.id, "anonymous");
        assert_eq!(anon.credential_type, CredentialType::None);
        assert_eq!(anon.authorization_level, AuthorizationLevel::ReadOnly);
    }

    #[test]
    fn test_authorization_levels() {
        let mut identity = AuthenticatedIdentity::anonymous();
        identity.authorization_level = AuthorizationLevel::Operator;

        assert!(identity.has_authorization(AuthorizationLevel::ReadOnly));
        assert!(identity.has_authorization(AuthorizationLevel::User));
        assert!(identity.has_authorization(AuthorizationLevel::Operator));
        assert!(!identity.has_authorization(AuthorizationLevel::Administrator));
    }

    #[test]
    fn test_api_key_authentication() {
        let mut provider = InMemoryAuthProvider::new();

        let identity = AuthenticatedIdentity {
            id: "test_user".to_string(),
            authorization_level: AuthorizationLevel::Operator,
            ..Default::default()
        };

        provider.register_api_key("secret_key_123", identity);

        // Valid key should authenticate
        let result = provider.authenticate_api_key("secret_key_123");
        assert!(result.success);
        assert_eq!(result.identity.unwrap().id, "test_user");

        // Invalid key should fail
        let result = provider.authenticate_api_key("wrong_key");
        assert!(!result.success);
    }

    #[test]
    fn test_credential_revocation() {
        let mut provider = InMemoryAuthProvider::new();

        let identity = AuthenticatedIdentity {
            id: "revoke_me".to_string(),
            authorization_level: AuthorizationLevel::User,
            ..Default::default()
        };

        provider.register_api_key("revoke_key", identity);

        // Should work before revocation
        let result = provider.authenticate_api_key("revoke_key");
        assert!(result.success);

        // Revoke
        provider.revoke("revoke_me");

        // Should fail after revocation
        let result = provider.authenticate_api_key("revoke_key");
        assert!(!result.success);
        assert!(matches!(result.error, Some(AuthenticationError::CredentialRevoked)));
    }
}
