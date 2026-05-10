#!/bin/bash
set -e

echo "========================================"
echo "EngineeringRAG - Linux Setup Script"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root or with sudo
check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        log_warn "This script should be run with sudo for system-wide installation"
        log_info "Example: sudo bash $0"
    fi
}

# --- 1. Update System ---
update_system() {
    log_info "Updating system packages..."
    if command -v apt-get &> /dev/null; then
        apt-get update -y
        apt-get upgrade -y
    elif command -v yum &> /dev/null; then
        yum update -y
    elif command -v dnf &> /dev/null; then
        dnf update -y
    else
        log_error "Package manager not found. Unsupported Linux distribution."
        exit 1
    fi
}

# --- 2. Install Basic Utilities ---
install_utilities() {
    log_info "Installing basic utilities..."

    local pkgs="curl wget git python3 python3-pip python3-venv build-essential lsb-release"

    if command -v apt-get &> /dev/null; then
        apt-get install -y $pkgs
    elif command -v yum &> /dev/null; then
        yum install -y $pkgs
    elif command -v dnf &> /dev/null; then
        dnf install -y $pkgs
    fi
}

# --- 3. Install Docker ---
install_docker() {
    log_info "Checking Docker installation..."

    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1)
        log_info "Docker $docker_version is already installed"

        # Check if user is in docker group
        if ! groups $(whoami) | grep -q docker; then
            log_warn "User $(whoami) is not in docker group"
            read -p "Add user to docker group? (y/N) " -n 1 -r
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                usermod -aG docker $(whoami)
                log_info "Added user to docker group. Log out and log back in to apply."
            fi
        fi
        return 0
    fi

    log_info "Installing Docker..."

    # Install Docker using official script
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    bash /tmp/get-docker.sh

    # Start and enable Docker service
    systemctl enable docker 2>/dev/null || true
    systemctl start docker 2>/dev/null || true

    # Add current user to docker group
    usermod -aG docker $(whoami)

    # Fix Docker permissions for existing containers
    log_info "Fixing Docker permissions..."
    chmod 666 /var/run/docker.sock 2>/dev/null || true

    log_info "Docker installed successfully"
}

# --- 4. Install NVIDIA Docker Toolkit ---
install_nvidia_docker() {
    log_info "Checking NVIDIA Docker Toolkit..."

    # Check if nvidia-container-toolkit is installed
    if command -v nvidia-ctk &> /dev/null || dpkg -l | grep -q nvidia-container-toolkit; then
        log_info "NVIDIA Docker Toolkit is already installed"
        # Configure Docker runtime if not already done
        if ! grep -q '"runtimes"' /etc/docker/daemon.json 2>/dev/null; then
            log_info "Configuring Docker runtime for NVIDIA..."
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker 2>/dev/null || true
        fi
        return 0
    fi

    log_info "Installing NVIDIA Docker Toolkit..."

    # Install required packages first
    apt-get update -y
    apt-get install -y curl gnupg

    # Add NVIDIA GPG key
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true

    # Use stable deb repository (works for Ubuntu 22.04, 24.04, and other Debian-based)
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Update and install
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit

    # Configure Docker runtime
    log_info "Configuring Docker runtime for NVIDIA..."
    nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true

    # Restart Docker to apply changes
    systemctl restart docker 2>/dev/null || true

    log_info "NVIDIA Docker Toolkit installed successfully"
}

# --- 5. Install CUDA 12.9 ---
install_cuda() {
    log_info "Checking CUDA 12.9 installation..."

    # Check if CUDA 12.9 is already installed
    if [ -d "/usr/local/cuda-12.9" ]; then
        local cuda_version=$(cat /usr/local/cuda-12.9/version.txt 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "unknown")
        log_info "CUDA $cuda_version is already installed at /usr/local/cuda-12.9"

        # Set environment variables
        export PATH=/usr/local/cuda-12.9/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

        if nvcc --version &> /dev/null; then
            nvcc --version
            return 0
        fi
    fi

    # Check if any CUDA is installed
    if [ -d "/usr/local/cuda" ]; then
        local current_version=$(cat /usr/local/cuda/version.txt 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "unknown")
        log_warn "CUDA $current_version is installed but CUDA 12.9 is required"
        log_info "Installing CUDA 12.9 alongside existing version..."
    fi

    # Check if NVIDIA drivers are installed
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "NVIDIA drivers not detected. Skipping CUDA toolkit installation."
        log_info "Install NVIDIA drivers first: https://www.nvidia.com/Download/index.aspx"
        return 0
    fi

    # CUDA 12.9 installation via apt (more reliable than .run)
    log_info "Installing CUDA 12.9 via apt..."

    # Add CUDA repository
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-12-9

    # Create symlink for /usr/local/cuda pointing to cuda-12.9
    rm -rf /usr/local/cuda
    ln -sf /usr/local/cuda-12.9 /usr/local/cuda

    # Set environment variables in system profile
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/environment

    # Export for current session
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    log_info "CUDA 12.9 installed successfully"
}

# --- 6. Install uv ---
install_uv() {
    log_info "Installing uv (Python package manager)..."

    if command -v uv &> /dev/null; then
        log_info "uv is already installed"
        return 0
    fi

    # Install uv via official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    log_info "uv installed successfully"
}

# --- 7. Create Directory Structure ---
create_directories() {
    log_info "Creating directory structure..."

    mkdir -p data/minio
    mkdir -p data/docling_models
    mkdir -p data/fastembed_cache
    mkdir -p data/huggingface_cache
    mkdir -p data/cache_airflow
    mkdir -p client-postgres-init

    # Set permissions (owner full, group read/write/execute, others read/execute)
    chmod -R 750 data/ 2>/dev/null || chmod -R 775 data/

    log_info "Directory structure created"
}

# --- 8. Verify GPU and CUDA Access ---
verify_cuda() {
    log_info "Verifying GPU and CUDA access..."

    # Check nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found. NVIDIA drivers may not be installed."
        log_info "GPU functionality will not be available until drivers are installed."
    else
        nvidia-smi
    fi

    # Check CUDA compiler
    if command -v nvcc &> /dev/null; then
        nvcc --version
    else
        log_warn "nvcc not found. CUDA toolkit not installed or not in PATH."
    fi

    # Test CUDA in Docker
    if command -v docker &> /dev/null; then
        if docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvcc --version 2>/dev/null; then
            log_info "CUDA access in Docker verified successfully!"
        else
            log_warn "CUDA in Docker test failed."
        fi
    else
        log_warn "Docker not installed. Cannot verify Docker CUDA access."
    fi
}

# --- 9. Fix Docker Image Permissions ---
fix_airflow_permissions() {
    log_info "Setting up Airflow permissions..."

    if [ -d "airflow" ]; then
        chmod -R 775 airflow/
        find airflow -type d -exec chmod g+s {} \;
        log_info "Airflow permissions fixed"
    else
        log_warn "airflow directory not found"
    fi
}

# --- Main Execution ---
main() {
    log_info "Starting EngineeringRAG setup..."
    log_info "This will install Docker, CUDA 12.9, and other dependencies"
    echo ""

    check_sudo

    # Run installation steps
    update_system
    install_utilities
    install_docker
    install_nvidia_docker
    install_cuda
    install_uv
    create_directories
    fix_airflow_permissions
    verify_cuda

    echo ""
    log_info "========================================"
    log_info "Setup completed successfully!"
    log_info "========================================"
    echo ""
    log_info "Next steps:"
    log_info "1. Log out and log back in (for Docker group changes)"
    log_info "2. Run: cd /home/dved/EngineeringRAG && docker compose up -d"
    echo ""
    log_info "Note: If you see GPU-related errors, run:"
    log_info "    sudo nvidia-ctk runtime configure --runtime=docker"
    log_info "    sudo systemctl restart docker"
    echo ""
}

main "$@"
