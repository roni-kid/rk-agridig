#!/bin/bash

################################################################################
# RK AgriDig — llama.cpp Compilation & Setup Script
#
# This script:
# 1. Clones llama.cpp repository
# 2. Compiles with CPU optimizations (OpenBLAS for matrix operations)
# 3. Installs llama-bench tool
# 4. Sets up Ollama for model serving
# 5. Verifies the installation
#
# Requirements:
# - Ubuntu 22.04 LTS (or similar Debian-based Linux)
# - GCC 11+ or Clang 14+
# - CMake 3.21+
# - git
#
# Usage:
#     bash setup.sh
#     bash setup.sh --skip-ollama    # Skip Ollama installation
#
################################################################################

set -e  # Exit on any error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LLAMACPP_DIR="${PROJECT_ROOT}/llama.cpp"
BUILD_DIR="${LLAMACPP_DIR}/build"
INSTALL_PREFIX="${PROJECT_ROOT}/local"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

log_info "================================"
log_info "RK AgriDig — Setup & Compilation"
log_info "================================"

log_info "Running pre-flight checks..."

# Check OS
if [[ ! -f /etc/os-release ]]; then
    log_error "Cannot determine OS. This script requires Linux."
    exit 1
fi

OS_NAME=$(grep "^NAME=" /etc/os-release | cut -d'"' -f2)
log_info "Detected OS: $OS_NAME"

# Check for required tools
for cmd in git cmake gcc g++ make; do
    if ! command -v $cmd &> /dev/null; then
        log_error "$cmd is not installed."
        echo "Install with: sudo apt-get install build-essential cmake git"
        exit 1
    fi
done

log_success "All required tools found"

# Check compiler version
GCC_VERSION=$(gcc --version | head -1 | awk '{print $NF}')
log_info "GCC version: $GCC_VERSION"

CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $NF}')
log_info "CMake version: $CMAKE_VERSION"

# ============================================================================
# Step 1: Install System Dependencies
# ============================================================================

log_info ""
log_info "Step 1/4: Installing system dependencies..."

# Check if running with sudo privileges (for apt-get)
if [[ $EUID -ne 0 ]]; then
    log_warn "Not running as root. Some system packages may require sudo."
    log_info "Installing optional dependencies (OpenBLAS for CPU acceleration)..."
    
    # Try to install without sudo (will likely fail, but worth trying)
    sudo apt-get update 2>/dev/null || log_warn "Could not update package list"
    sudo apt-get install -y libopenblas-dev libomp-dev 2>/dev/null || \
        log_warn "Could not install optional CPU acceleration libraries"
else
    log_info "Installing dependencies..."
    apt-get update
    apt-get install -y \
        build-essential \
        cmake \
        git \
        libopenblas-dev \
        libomp-dev \
        pkg-config
    log_success "System dependencies installed"
fi

# ============================================================================
# Step 2: Clone & Compile llama.cpp
# ============================================================================

log_info ""
log_info "Step 2/4: Cloning and compiling llama.cpp..."

if [[ -d "$LLAMACPP_DIR" ]]; then
    log_warn "llama.cpp already cloned at $LLAMACPP_DIR"
    log_info "Updating repository..."
    cd "$LLAMACPP_DIR"
    git pull origin master || log_warn "Could not update repository"
else
    log_info "Cloning llama.cpp from GitHub..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMACPP_DIR"
    cd "$LLAMACPP_DIR"
    log_success "Repository cloned"
fi

# Create build directory
if [[ -d "$BUILD_DIR" ]]; then
    log_warn "Build directory already exists. Cleaning..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

log_info "Configuring CMake with CPU optimizations..."

# CMake configuration with:
# - CPU-only inference (no CUDA)
# - OpenBLAS for matrix operations (significant speedup)
# - Release mode for performance
# - Optimizations enabled
# NOTE: current llama.cpp CMake options use the GGML_* prefix, not LLAMA_*.
# The old LLAMA_BLAS / LLAMA_NATIVE / LLAMA_F16C / LLAMA_FMA names were
# renamed upstream; passing them here is silently ignored by CMake (unknown
# -D flags don't error by default), which means OpenBLAS was never actually
# being enabled despite this step reporting success.
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_NATIVE=ON \
    .. || {
    log_error "CMake configuration failed"
    log_info "Trying without OpenBLAS (CPU will be slower)..."
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DGGML_NATIVE=ON \
        ..
}

log_info "Compiling llama.cpp (this may take 5–15 minutes)..."
NPROC=$(nproc)
log_info "Using $NPROC CPU cores for compilation"

make -j "$NPROC" || {
    log_error "Compilation failed. Check errors above."
    exit 1
}

log_success "llama.cpp compiled successfully"

# Install binaries
log_info "Installing binaries to $INSTALL_PREFIX..."
make install

log_success "Installation complete"

# ============================================================================
# Step 3: Build llama-bench (Benchmarking Tool)
# ============================================================================

log_info ""
log_info "Step 3/4: Building llama-bench..."

cd "$LLAMACPP_DIR/build"

# Compile llama-bench
if cmake --build . --target llama-bench 2>/dev/null; then
    log_success "llama-bench built successfully"
else
    log_warn "Could not build llama-bench as target. Using default build output."
fi

# Create symlink to llama-bench if it exists
if [[ -f "$BUILD_DIR/bin/llama-bench" ]]; then
    ln -sf "$BUILD_DIR/bin/llama-bench" "$INSTALL_PREFIX/bin/llama-bench" 2>/dev/null || true
    log_success "llama-bench symlinked"
elif [[ -f "$BUILD_DIR/llama-bench" ]]; then
    ln -sf "$BUILD_DIR/llama-bench" "$INSTALL_PREFIX/bin/llama-bench" 2>/dev/null || true
    log_success "llama-bench found"
else
    log_warn "llama-bench binary not found. You may need to build it manually."
fi

# ============================================================================
# Step 4: Setup Ollama (Optional)
# ============================================================================

log_info ""
log_info "Step 4/4: Setting up Ollama..."

SKIP_OLLAMA=false
if [[ "$1" == "--skip-ollama" ]]; then
    SKIP_OLLAMA=true
fi

if [[ "$SKIP_OLLAMA" == true ]]; then
    log_info "Skipping Ollama installation (--skip-ollama flag)"
else
    if command -v ollama &> /dev/null; then
        log_success "Ollama is already installed"
        ollama --version
    else
        log_info "Ollama not found. Installing..."
        
        # Download and install Ollama
        if curl -fsSL https://ollama.ai/install.sh | sh 2>/dev/null; then
            log_success "Ollama installed successfully"
        else
            log_warn "Could not install Ollama automatically."
            log_info "You can install manually at: https://ollama.ai"
        fi
    fi
fi

# ============================================================================
# Verification & Summary
# ============================================================================

log_info ""
log_info "Verifying installation..."

# Check llama-cli exists (llama.cpp renamed the CLI binary from 'main' to
# 'llama-cli'; the old name no longer exists in current builds)
if [[ -f "$BUILD_DIR/bin/llama-cli" ]] || [[ -f "$BUILD_DIR/llama-cli" ]]; then
    log_success "llama-cli found"
else
    log_warn "Could not locate llama-cli binary"
fi

# Check llama-bench
if command -v llama-bench &> /dev/null || [[ -f "$BUILD_DIR/llama-bench" ]]; then
    log_success "llama-bench found"
else
    log_warn "llama-bench not found in PATH"
fi

# ============================================================================
# Final Summary
# ============================================================================

log_info ""
log_info "=========================================="
log_success "SETUP COMPLETE"
log_info "=========================================="

echo ""
echo "Next steps:"
echo ""
echo "1. Download the model:"
echo "   python models/download_model.py"
echo ""
echo "2. Run benchmarks:"
echo "   bash benchmarks/run_profiler.sh"
echo ""
echo "3. Start Ollama server (in background):"
echo "   ollama serve &"
echo ""
echo "4. Launch the Gradio UI:"
echo "   python ui/app.py"
echo ""
echo "For more details, see:"
echo "  - docs/SETUP.md"
echo "  - docs/USAGE.md"
echo ""
echo "Build directory: $BUILD_DIR"
echo "Install prefix:  $INSTALL_PREFIX"
echo ""

# Create a marker file to indicate successful setup
touch "${PROJECT_ROOT}/.setup-complete"

log_success "All systems ready for Phase 1!"