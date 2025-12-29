# Build Instructions for VolumeRover Qt6

This guide covers building VolumeRover with Qt6 on Windows and Linux.

## Prerequisites

### Common Requirements
- CMake 3.16 or later
- Git
- C++17 compatible compiler

### Platform-Specific Requirements

#### Windows
- Visual Studio 2019 or 2022 (Community Edition is fine)
  - Install "Desktop development with C++" workload
- Qt6 (6.5 LTS or later recommended)
  - Download from: https://www.qt.io/download-open-source-qt
  - Install components: Qt 6.x for MSVC 2019/2022 (64-bit), Qt Creator (optional)

#### Linux (Ubuntu 22.04 / Debian 12)
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    qt6-base-dev \
    qt6-base-dev-tools \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libboost-all-dev \
    libglew-dev \
    freeglut3-dev \
    libfftw3-dev \
    libmagick++-dev \
    imagemagick
```

**Note:** ImageMagick (libmagick++-dev) is required for some volume utilities that process image files.

#### Linux (Fedora 38+)
```bash
sudo dnf install -y \
    gcc-c++ \
    cmake \
    git \
    qt6-qtbase-devel \
    qt6-qttools-devel \
    boost-devel \
    glew-devel \
    freeglut-devel \
    fftw-devel \
    ImageMagick-devel \
    ImageMagick
```

**Note:** ImageMagick-devel is required for some volume utilities that process image files.

### Optional Dependencies

#### For CGAL-based features
**Ubuntu/Debian:**
```bash
sudo apt-get install libcgal-dev libgmp-dev libmpfr-dev
```

**Fedora:**
```bash
sudo dnf install CGAL-devel gmp-devel mpfr-devel
```

**Windows:**
- Download CGAL from https://www.cgal.org/
- Extract and set CGAL_DIR environment variable

#### For GSL (GNU Scientific Library)
**Ubuntu/Debian:**
```bash
sudo apt-get install libgsl-dev
```

**Fedora:**
```bash
sudo dnf install gsl-devel
```

#### For CUDA (GPU acceleration - optional)
**Ubuntu/Debian:**
```bash
# NVIDIA drivers must be installed first
sudo apt-get install nvidia-cuda-toolkit
```

**Windows:**
- Download CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
- Follow installer instructions

#### For ImageMagick (volume utilities - optional on Windows)
**Windows:**
- Download from https://imagemagick.org/script/download.php
- Install the "ImageMagick-7.x.x-Q16-HDRI-x64-dll.exe" version
- Add to PATH during installation

**Note:** ImageMagick is optional but recommended for volume processing utilities that handle image formats.

## Building from Source

### 1. Clone the Repository

```bash
git clone https://github.com/transfix/volrover.git
cd volrover
```

### 2. Configure Build with CMake

#### Linux

```bash
# Create build directory
mkdir build
cd build

# Configure with Qt6
cmake .. -DUSE_QT6=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local

# Optional: Disable CGAL if not installed
# cmake .. -DUSE_QT6=ON -DDISABLE_CGAL=ON -DCMAKE_BUILD_TYPE=Release
```

#### Windows (Visual Studio)

Open "x64 Native Tools Command Prompt for VS 2022" (or 2019), then:

```cmd
REM Set Qt6 path (adjust to your installation)
set Qt6_DIR=C:\Qt\6.5.3\msvc2019_64

REM Create build directory
mkdir build
cd build

REM Configure
cmake .. -G "Visual Studio 17 2022" -A x64 ^
         -DUSE_QT6=ON ^
         -DQt6_DIR=%Qt6_DIR% ^
         -DCMAKE_PREFIX_PATH=%Qt6_DIR%

REM Or for MinGW:
REM cmake .. -G "MinGW Makefiles" -DUSE_QT6=ON -DQt6_DIR=%Qt6_DIR%
```

#### macOS

```bash
# Install dependencies via Homebrew first
brew install cmake qt@6 boost glew fftw

# Create build directory
mkdir build
cd build

# Configure
cmake .. -DUSE_QT6=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/qt@6
```

### 3. Build

#### Linux / macOS
```bash
# Use all CPU cores
cmake --build . -j$(nproc)

# Or with make directly:
make -j$(nproc)
```

#### Windows
```cmd
REM Visual Studio
cmake --build . --config Release -j8

REM Or open VolumeRover.sln in Visual Studio and build from IDE
```

### 4. Install (Optional)

#### Linux
```bash
sudo cmake --install .
```

#### Windows
```cmd
cmake --install . --config Release
```

## Build Options

Configure build with CMake options:

```bash
# Enable/disable Qt6
cmake .. -DUSE_QT6=ON

# Disable CGAL
cmake .. -DDISABLE_CGAL=ON

# Enable optional modules
cmake .. -DBUILD_TILING_LIB=ON
cmake .. -DBUILD_SEGMENTATION_LIB=ON
cmake .. -DBUILD_VOLUMEGRIDROVER=ON

# Enable CUDA features
cmake .. -DBUILD_MSLEVELSET_LIB=ON
cmake .. -DBUILD_HOSEGMENTATION_LIB=ON

# Enable stereo display
cmake .. -DENABLE_STEREO_DISPLAY=ON

# Set installation prefix
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/volumerover
```

## Creating Binary Packages

### Linux (Debian Package)

```bash
# After building
cd build
cpack -G DEB

# This creates VolumeRover-2.0.0-Linux-x86_64.deb
sudo dpkg -i VolumeRover-2.0.0-Linux-x86_64.deb
```

### Linux (AppImage)

```bash
# Install linuxdeploy
wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
wget https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage
chmod +x linux*.AppImage

# Create AppImage
cmake --install . --prefix AppDir
./linuxdeploy-x86_64.AppImage --appdir AppDir --plugin qt --output appimage
```

### Windows (NSIS Installer)

```cmd
REM Install NSIS first: https://nsis.sourceforge.io/

REM After building
cd build
cpack -G NSIS

REM This creates VolumeRover-2.0.0-win64-x64.exe installer
```

### Windows (Portable ZIP)

```cmd
REM Use windeployqt to bundle Qt libraries
cd build\bin\Release

REM Assuming Qt6 is in PATH
windeployqt VolumeRover2.exe

REM Create ZIP
cd ..\..
powershell Compress-Archive -Path bin\Release\* -DestinationPath VolumeRover-2.0.0-win64-portable.zip
```

## Running VolumeRover

### Linux
```bash
# If installed system-wide
VolumeRover2

# Or from build directory
./bin/VolumeRover2
```

### Windows
```cmd
REM From build directory
bin\Release\VolumeRover2.exe

REM Or double-click the executable in File Explorer
```

### macOS
```bash
# From build directory
./bin/VolumeRover2.app/Contents/MacOS/VolumeRover2

# Or open the app bundle
open bin/VolumeRover2.app
```

## Important Dependency Changes

### GLEW (OpenGL Extension Wrangler)

As of the Qt6 migration, VolumeRover now uses the **system-installed GLEW library** instead of a bundled version. This change was made because:

- The bundled GLEW version was very old and unmaintained
- System package managers provide up-to-date, security-patched versions
- Reduces maintenance burden and binary size

**Requirements:**
- **Linux:** Install `libglew-dev` (Ubuntu/Debian) or `glew-devel` (Fedora/RHEL)
- **Windows:** GLEW is typically bundled with graphics SDKs or can be obtained from http://glew.sourceforge.net/
- **macOS:** Install via Homebrew: `brew install glew`

**Ubuntu/Debian:**
```bash
sudo apt-get install libglew-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install glew-devel
```

**Note:** The old bundled GLEW source in `src/glew/` and `inc/glew/` is no longer compiled but remains in the repository for reference. These directories may be removed in future versions.

## Troubleshooting

### Qt6 Not Found

**Linux:**
- Ensure qt6-base-dev is installed
- Set CMAKE_PREFIX_PATH: `cmake .. -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/Qt6`

**Windows:**
- Set Qt6_DIR: `cmake .. -DQt6_DIR=C:\Qt\6.5.3\msvc2019_64`
- Or add Qt to PATH before running cmake

### OpenGL Issues

**Linux:**
- Install Mesa drivers: `sudo apt-get install libgl1-mesa-dev`
- For NVIDIA: Install proprietary drivers

**Windows:**
- Update graphics drivers
- If using remote desktop, OpenGL may be limited

### CGAL Compilation Errors

```bash
# Try disabling CGAL
cmake .. -DDISABLE_CGAL=ON
```

### Boost Not Found

**Linux:**
```bash
sudo apt-get install libboost-all-dev
```

**Windows:**
- Download prebuilt binaries from https://sourceforge.net/projects/boost/files/boost-binaries/
- Or set BOOST_ROOT: `cmake .. -DBOOST_ROOT=C:\boost_1_83_0`

### CUDA Build Failures

- Ensure CUDA Toolkit matches your GCC/MSVC version
- Check CUDA compatibility matrix: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

### Missing Qt Plugins at Runtime

**Linux:**
```bash
# Install Qt platform plugins
sudo apt-get install qt6-qpa-plugins
```

**Windows:**
- Run windeployqt on the executable
- Or manually copy platform plugins from Qt installation

## Build Performance Tips

- Use Ninja generator for faster builds: `cmake .. -G Ninja`
- Use ccache on Linux: `sudo apt-get install ccache`
- On Windows, use `/MP` flag (already set in CMake)
- Build on SSD/NVMe storage for faster I/O

## Getting Help

- Check existing issues: https://github.com/transfix/volrover/issues
- Review Qt6 migration plan: `QT6_MIGRATION_PLAN.md`
- Qt6 documentation: https://doc.qt.io/qt-6/

## Minimum System Requirements

### Runtime
- OS: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- RAM: 4 GB minimum, 8 GB recommended
- GPU: OpenGL 3.3+ compatible graphics card
- Display: 1280x720 minimum resolution

### Build
- RAM: 8 GB minimum, 16 GB recommended
- Disk Space: 10 GB free for build artifacts
- CPU: Multi-core processor recommended for parallel builds
