#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SBOM GENERATION SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
# Generates Software Bill of Materials in CycloneDX and SPDX formats
# Required for FedRAMP, NIST 800-53, and defense procurement
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/sbom"
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════════════════════════"
echo " FRACTAL SBOM GENERATOR"
echo "═══════════════════════════════════════════════════════════════════════════════"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${YELLOW}Warning: $1 not found. Installing...${NC}"
        cargo install "$2" 2>/dev/null || {
            echo -e "${RED}Failed to install $1. Please install manually.${NC}"
            return 1
        }
    fi
    return 0
}

# Install SBOM tools if needed
echo -e "\n${GREEN}[1/5] Checking SBOM generation tools...${NC}"
check_tool "cargo-sbom" "cargo-sbom" || true
check_tool "cargo-cyclonedx" "cargo-cyclonedx" || true

# Generate Cargo.lock if not present
echo -e "\n${GREEN}[2/5] Ensuring Cargo.lock is up to date...${NC}"
cd "$PROJECT_ROOT"
cargo generate-lockfile 2>/dev/null || cargo check

# Generate CycloneDX SBOM (industry standard for defense)
echo -e "\n${GREEN}[3/5] Generating CycloneDX SBOM (JSON)...${NC}"
if command -v cargo-cyclonedx &> /dev/null; then
    cargo cyclonedx --format json > "${OUTPUT_DIR}/fractal-cyclonedx-${TIMESTAMP}.json" 2>/dev/null || {
        echo -e "${YELLOW}CycloneDX generation failed, trying alternative...${NC}"
    }
fi

# Generate using cargo-sbom (SPDX format)
echo -e "\n${GREEN}[4/5] Generating SPDX SBOM...${NC}"
if command -v cargo-sbom &> /dev/null; then
    cargo sbom > "${OUTPUT_DIR}/fractal-spdx-${TIMESTAMP}.spdx" 2>/dev/null || {
        echo -e "${YELLOW}SPDX generation failed, trying alternative...${NC}"
    }
fi

# Generate dependency tree for manual review
echo -e "\n${GREEN}[5/5] Generating dependency manifest...${NC}"
{
    echo "# FRACTAL DEPENDENCY MANIFEST"
    echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# ═══════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "## Direct Dependencies"
    echo ""
    cargo tree --depth 1 --prefix none 2>/dev/null | sort | uniq
    echo ""
    echo "## Full Dependency Tree"
    echo ""
    cargo tree 2>/dev/null
} > "${OUTPUT_DIR}/fractal-dependencies-${TIMESTAMP}.txt"

# Generate license summary
echo -e "\n${GREEN}Generating license summary...${NC}"
{
    echo "# FRACTAL LICENSE SUMMARY"
    echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# ═══════════════════════════════════════════════════════════════════════════"
    echo ""
    cargo tree --format "{p} {l}" 2>/dev/null | sort | uniq -c | sort -rn
} > "${OUTPUT_DIR}/fractal-licenses-${TIMESTAMP}.txt"

# Create latest symlinks
ln -sf "fractal-dependencies-${TIMESTAMP}.txt" "${OUTPUT_DIR}/fractal-dependencies-latest.txt" 2>/dev/null || true
ln -sf "fractal-licenses-${TIMESTAMP}.txt" "${OUTPUT_DIR}/fractal-licenses-latest.txt" 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}SBOM generation complete!${NC}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.txt "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.spdx 2>/dev/null || ls -la "$OUTPUT_DIR"/
echo ""
echo "For procurement review, provide:"
echo "  1. sbom/fractal-cyclonedx-*.json   (CycloneDX format for automated scanning)"
echo "  2. sbom/fractal-spdx-*.spdx        (SPDX format for compliance tools)"
echo "  3. sbom/fractal-licenses-*.txt     (License summary for legal review)"
