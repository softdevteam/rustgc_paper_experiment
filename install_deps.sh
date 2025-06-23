#!/bin/sh

set -e

fail() {
    echo "Error: $1" 1>&2
    exit 1
}

# Check for --no-cpu-gov flag
NO_CPU_GOV=0
if [ "$1" = "--no-cpu-gov" ]; then
    NO_CPU_GOV=1
fi

# Detect distro and set commands
if [ -f /etc/debian_version ]; then
    DISTRO=debian
    PKG_FILE="packages-debian.txt"
elif [ -f /etc/arch-release ]; then
    DISTRO=arch
    PKG_FILE="packages-arch.txt"
else
    fail "Unsupported or unrecognized Linux distribution."
fi

# Check for package file
if [ ! -f "$PKG_FILE" ]; then
    fail "Package list file '$PKG_FILE' not found."
fi

echo "Detected Linux distribution: $DISTRO"
echo "Checking for missing packages..."

# Read package list
PKGS=""
for pkg in $(grep -v '^[[:space:]]*#' "$PKG_FILE" | grep -v '^[[:space:]]*$'); do
    PKGS="$PKGS $pkg"
done

MISSING_PKGS=""
for pkg in $PKGS; do
    if [ "$DISTRO" = "debian" ]; then
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            MISSING_PKGS="$MISSING_PKGS $pkg"
        fi
    elif [ "$DISTRO" = "arch" ]; then
        if ! pacman -Q "$pkg" >/dev/null 2>&1; then
            MISSING_PKGS="$MISSING_PKGS $pkg"
        fi
    fi
done

if [ -z "$MISSING_PKGS" ]; then
    echo "All listed packages are already installed."
else
    echo "The following packages are missing and will be installed:"
    for pkg in $MISSING_PKGS; do
        echo "  $pkg"
    done
    echo

    # Install missing packages
    if [ "$DISTRO" = "debian" ]; then
        sudo apt-get update
        echo "$MISSING_PKGS" | xargs sudo apt-get install -y
    elif [ "$DISTRO" = "arch" ]; then
        sudo pacman -Sy --needed --noconfirm $MISSING_PKGS
    fi
fi

echo "System dependencies installed successfully."

# CPU frequency governor logic
if [ "$NO_CPU_GOV" -eq 0 ] && [ -d /sys/devices/system/cpu ]; then
    echo "Do you want to set the CPU frequency governor to 'performance'? [y/N]"
    printf "> "
    read ans
    case "$ans" in
        y|Y|yes|YES)
            echo "Setting CPU frequency governor to performance..."
            for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                if [ -f "$cpu" ]; then
                    echo "performance" | sudo tee "$cpu" >/dev/null
                fi
            done
            echo "CPU governor set to performance."
            ;;
        *)
            echo "Skipping CPU governor change."
            ;;
    esac
else
    echo "Skipping CPU governor change."
fi

# Install rustup and Rust nightly
if ! command -v rustup >/dev/null 2>&1; then
    echo "rustup not found. Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- --profile minimal --default-toolchain nightly -y
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
else
    echo "rustup already installed."
fi

echo "Installing and setting Rust nightly as default..."
rustup toolchain install nightly
rustup default nightly

echo "Rust nightly installed and set as default."
echo "All dependencies installed successfully."

echo
echo "===================================================================="
echo "IMPORTANT:"
echo "Before running experiments, you must have \$HOME/.cargo/bin in your PATH."
echo "You can add this to your shell profile with:"
echo '    export PATH="$HOME/.cargo/bin:$PATH"'
echo
echo "If you encounter issues, also check that \$HOME/.rustup exists."
echo "===================================================================="
echo

