# VolumeRover Qt6 Migration Plan

## Executive Summary

This document outlines the comprehensive plan for migrating VolumeRover from Qt4 to Qt6. VolumeRover is a scientific visualization and computational geometry application originally developed at the Computational Visualization Center at UT Austin.

**Current State**: Qt4 with Qt3Support compatibility layer  
**Target State**: Qt6 (latest stable version)  
**Platforms**: Windows and Linux  
**Build System**: CMake  

---

## 1. Current Dependencies

### Required Core Dependencies

#### Graphics and UI
- **Qt4** (QtCore, QtGui, QtXml, QtOpenGL, Qt3Support)
  - Currently using deprecated Qt3Support module
  - OpenGL integration via QtOpenGL
  
- **OpenGL/GLEW**
  - Used for 3D rendering
  - GLEW library for OpenGL extension loading
  - ~~Static build on Windows~~ **[UPDATED]** Now uses system GLEW library
  - **Migration Note**: Changed from bundled to system dependency (Dec 2025)

- **QGLViewer**
  - 3D viewer widget library (bundled with project)
  - Built on top of Qt OpenGL
  - Has VRender vectorial renderer component (optional)

#### Math and Scientific Computing
- **Boost** (>=1.34.0)
  - Used throughout codebase
  - Typically multithreaded, shared libraries

- **CGAL** (Computational Geometry Algorithms Library)
  - Optional but used by many modules:
    - Tiling/ContourTiler
    - Segmentation
    - Curation
    - PocketTunnel
    - Skeletonization
    - TightCocone
    - SuperSecondaryStructures
    - VolumeGridRover
  - Can be disabled with DISABLE_CGAL flag

- **FFTW** (Fast Fourier Transform Library)
  - Required for signal processing operations

- **GSL** (GNU Scientific Library)
  - Used for mathematical computations
  - Required by VolumeGridRover

- **LAPACK**
  - Linear algebra operations

#### Optional/Advanced Dependencies
- **CUDA**
  - Required for GPU-accelerated modules:
    - MSLevelSet
    - HigherOrderSegmentation (HOSegmentation)
    - MultiphaseSegmentation (MPSegmentation)
  - Only built when explicitly enabled

- **Log4cplus**
  - Logging library (bundled)

- **XmlRPC**
  - RPC communication (bundled)

- **PETSc** (Portable, Extensible Toolkit for Scientific Computation)
  - Optional, for advanced numerical computations

#### Platform-Specific
- **GLUT** (OpenGL Utility Toolkit)
  - Sometimes required on certain Qt configurations

- **Cg** (NVIDIA Cg Toolkit)
  - Used for shader programming in volume rendering
  - Required for fixup_bundle on some platforms

### Optional Module Dependencies

Many modules are disabled by default and have their own dependency chains:
- Tiling library → CGAL
- Segmentation → CGAL
- Secondary structures → CGAL, Histogram
- MMHLS → HLevelSet
- HOSegmentation → MSLevelSet, CUDA
- MPSegmentation → MSLevelSet, HOSegmentation, CUDA

---

## 2. Key Challenges in Qt6 Migration

### 2.1 Qt3Support Removal
**Impact**: HIGH  
**Issue**: Qt3Support module was deprecated in Qt4 and completely removed in Qt5+

**Files affected**:
- Many UI forms use Qt3Support widgets (Q3ButtonGroup, Q3GroupBox, Q3ProgressDialog)
- Header includes reference Qt3 classes
- Some dialogs in `zzzzzUNUSEDzzzzz/` folder still reference Qt3

**Solution**:
- Remove all Qt3Support dependencies
- Replace Q3 widgets with Qt5/Qt6 equivalents
- Most of this was likely already done for Qt4, but need to verify

### 2.2 QtOpenGL → OpenGLWidget Migration
**Impact**: HIGH  
**Issue**: QtOpenGL module is deprecated in Qt6, replaced by Qt OpenGL widgets in QtOpenGLWidgets

**Changes needed**:
- QGLWidget → QOpenGLWidget
- QGLFormat → QSurfaceFormat
- QGL* classes → QOpenGL* classes
- Update QGLViewer library to be compatible

### 2.3 CMake Build System Updates
**Impact**: MEDIUM  
**Issue**: Qt4/Qt5/Qt6 have different CMake integration patterns

**Current issues**:
- Uses old `qt4_wrap_cpp()` and `qt4_wrap_ui()` macros
- Uses manual MOC header specification
- Checks for QT3_FOUND and QT4_FOUND

**Solution**:
- Switch to Qt6's modern CMake system using `find_package(Qt6 ...)`
- Use `qt_wrap_cpp()`, `qt_wrap_ui()` (Qt6 provides non-versioned commands)
- Use automatic MOC/UIC handling with `AUTOMOC` and `AUTOUIC`
- Use `target_link_libraries()` with Qt6:: namespace

### 2.4 Header Includes
**Impact**: MEDIUM  
**Issue**: Qt header naming conventions changed

**Changes needed**:
- Lowercase Qt3/Qt4 headers (e.g., `qapplication.h`) → Title case (e.g., `QApplication`)
- Some headers split into separate modules
- Qt6 modules are more granular

### 2.5 QMake → CMake
**Impact**: LOW  
**Issue**: Project has many .pro files (QMake), but also has CMake files

**Solution**:
- Continue using CMake as primary build system
- Remove or ignore .pro files (appear to be legacy)

### 2.6 UI Forms
**Impact**: MEDIUM  
**Issue**: Separate Qt3 and Qt4 .ui forms exist

**Current structure**:
- `*.Qt3.ui` files (legacy)
- `*.Qt4.ui` files (current)

**Solution**:
- Qt6 can generally read Qt4 .ui files
- May need minor adjustments in Designer
- Test all dialogs after migration

---

## 3. Migration Strategy

### Phase 1: Preparation and Environment Setup
**Duration**: 1-2 weeks

1. **Set up Qt6 development environment**
   - Install Qt6 on Windows and Linux
   - Install Qt Creator (optional but helpful)
   - Verify all non-Qt dependencies are available for both platforms

2. **Create migration branch**
   - Branch from current master
   - Set up parallel build environments (Qt4 and Qt6)

3. **Audit codebase**
   - Inventory all Qt4-specific code
   - Identify Qt3Support usage
   - Document custom Qt extensions

4. **Dependency verification**
   - Test all dependencies build with modern compilers
   - Update CGAL, Boost, etc. if needed
   - Verify CUDA toolkit compatibility (if using GPU features)

### Phase 2: CMake Build System Modernization
**Duration**: 2-3 weeks

1. **Update root CMakeLists.txt**
   - Bump minimum CMake version to 3.16+ (required for Qt6)
   - Update project setup
   - Modernize compiler flags and options

2. **Update SetupQt.cmake**
   - Replace Qt4 find_package logic
   - Implement Qt6 component discovery
   - Remove Qt3 legacy code
   - Update include directories and definitions

3. **Update module CMakeLists.txt files**
   - Replace qt4_wrap_cpp/qt4_wrap_ui with modern equivalents
   - Enable AUTOMOC and AUTOUIC
   - Update link targets to Qt6:: namespace
   - Test each module independently

4. **Update QGLViewer**
   - Migrate to QOpenGLWidget
   - Update MOC handling
   - Test rendering functionality

### Phase 3: Source Code Migration
**Duration**: 4-6 weeks

1. **Update header includes**
   - Convert all lowercase Qt headers to title case
   - Add required Qt6 module includes
   - Fix split module headers (e.g., QtCore vs QtConcurrent)

2. **Remove Qt3Support**
   - Verify all Qt3Support usage is already removed or unused
   - Clean up conditional Qt3 code blocks
   - Remove Qt3 .ui forms

3. **OpenGL modernization**
   - QGLWidget → QOpenGLWidget throughout
   - QGLFormat → QSurfaceFormat
   - Update QGLViewer library
   - Test all 3D rendering paths

4. **Handle Qt6 API changes**
   - QString → QStringView where appropriate
   - QList<T> unification (was QVector<T>)
   - QRegExp → QRegularExpression
   - Deprecated signal/slot syntax updates
   - QVariant and QMetaType changes

5. **Update UI forms**
   - Load each .Qt4.ui in Qt Designer 6
   - Fix deprecated widgets
   - Verify layouts and properties
   - Test runtime behavior

### Phase 4: Core Modules Migration
**Duration**: 3-4 weeks

Priority order:
1. **CVC** (core library)
2. **VolMagick** (volume data handling)
3. **ColorTable2** (color management)
4. **QGLViewer** (3D viewer)
5. **VolumeRenderer** / **GeometryRenderer**
6. **VolumeRover2** (main application)

For each module:
- Update CMakeLists.txt
- Update source files
- Build and test independently
- Fix compilation errors
- Test functionality

### Phase 5: Optional Modules
**Duration**: 2-4 weeks

Migrate optional modules in dependency order:
- Tiling/ContourTiler
- Segmentation
- LBIE
- FastContouring
- HLevelSet
- MSLevelSet (if CUDA available)
- Other specialized modules

### Phase 6: Testing and Validation
**Duration**: 2-3 weeks

1. **Functional testing**
   - Load various dataset formats
   - Test volume rendering
   - Test geometry operations
   - Test all dialog boxes
   - Verify file I/O

2. **Platform testing**
   - Test on Windows 10/11
   - Test on Linux (Ubuntu, Fedora)
   - Verify graphics driver compatibility
   - Test stereo display (if enabled)

3. **Performance validation**
   - Compare rendering performance Qt4 vs Qt6
   - Memory usage analysis
   - GPU utilization testing

4. **Regression testing**
   - Ensure all Qt4 workflows still work
   - Verify scientific accuracy of computations
   - Test interoperability with data files

### Phase 7: Binary Distribution Setup
**Duration**: 2-3 weeks

1. **Windows installer**
   - Update CPack configuration
   - Include Qt6 DLLs and plugins
   - Test NSIS installer
   - Handle Visual C++ redistributables
   - Include OpenGL libraries
   - Code signing (if available)

2. **Linux packages**
   - Create AppImage
   - Debian/Ubuntu .deb package
   - RPM for Fedora/RHEL
   - Flatpak or Snap (optional)
   - Handle Qt6 library bundling
   - Desktop integration files

3. **Documentation**
   - Update build instructions
   - Document dependency installation
   - Create user installation guide
   - Troubleshooting guide

---

## 4. Detailed CMakeLists.txt Changes

### 4.1 Root CMakeLists.txt

**Before (Qt4)**:
```cmake
CMAKE_MINIMUM_REQUIRED (VERSION 2.6.2)
PROJECT (VolumeRover)

include( SetupQt )
```

**After (Qt6)**:
```cmake
cmake_minimum_required(VERSION 3.16)
project(VolumeRover VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable automatic MOC, UIC, RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

include(SetupQt)
```

### 4.2 SetupQt.cmake

**Complete replacement**:
```cmake
#
# Modern Qt6 setup macro
#
macro(SetupQt)
  # Find Qt6
  find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Xml
    OpenGL
    OpenGLWidgets
  )
  
  # Qt6 automatically sets up include directories and compiler flags
  # No need for manual QT_USE_* variables
  
  # For Windows static builds
  if(WIN32)
    if(Qt6_IS_STATIC)
      add_definitions(-DQT_STATICPLUGIN)
    endif()
  endif(WIN32)
  
  # Clean namespace definition
  add_definitions(-DQT_CLEAN_NAMESPACE)
  
endmacro(SetupQt)
```

### 4.3 Module CMakeLists.txt Updates

**Before (Qt4)**:
```cmake
SetupQt()

SET(MOC_HEADERS
  ../../inc/VolumeRover2/CVCMainWindow.h
)

qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
qt4_wrap_ui(UI_FILES ${UI4_FORMS})

add_executable(VolumeRover2 
  ${SOURCE_FILES}
  ${MOC_SOURCES}
  ${UI_FILES}
)

target_link_libraries(VolumeRover2 ${QT_LIBRARIES})
```

**After (Qt6 - with AUTOMOC)**:
```cmake
SetupQt()

# UI files location for AUTOUIC
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(VolumeRover2 
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${UI_FORMS}  # AUTOUIC will handle these
)

target_link_libraries(VolumeRover2 
  PRIVATE
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::Xml
  Qt6::OpenGL
  Qt6::OpenGLWidgets
  ${LIBS}
)
```

### 4.4 QGLViewer Module Updates

Major changes needed:
- Replace QGLWidget base class with QOpenGLWidget
- Update context creation
- Handle OpenGL function pointer resolution differently

---

## 5. Code Migration Patterns

### 5.1 Header Updates

**Before**:
```cpp
#include <qapplication.h>
#include <qprogressdialog.h>
#include <qevent.h>
```

**After**:
```cpp
#include <QApplication>
#include <QProgressDialog>
#include <QEvent>
```

### 5.2 OpenGL Widget Migration

**Before (Qt4)**:
```cpp
#include <QGLWidget>

class MyViewer : public QGLWidget {
    Q_OBJECT
public:
    MyViewer(QWidget* parent = 0);
protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int w, int h);
};
```

**After (Qt6)**:
```cpp
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

class MyViewer : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    MyViewer(QWidget* parent = nullptr);
protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
};
```

### 5.3 QRegExp → QRegularExpression

**Before**:
```cpp
#include <QRegExp>

QRegExp rx("\\d+");
if (rx.exactMatch(text)) {
    // ...
}
```

**After**:
```cpp
#include <QRegularExpression>

QRegularExpression rx("\\d+");
QRegularExpressionMatch match = rx.match(text);
if (match.hasMatch()) {
    // ...
}
```

### 5.4 Signal/Slot Connection

**Before (old style)**:
```cpp
connect(button, SIGNAL(clicked()), this, SLOT(onButtonClicked()));
```

**After (Qt6 - modern syntax)**:
```cpp
connect(button, &QPushButton::clicked, this, &MyClass::onButtonClicked);
```

---

## 6. Installation Requirements

### 6.1 Build-time Dependencies

#### Windows
```
Required:
- Qt6 (6.5+) with components: Core, Gui, Widgets, Xml, OpenGL, OpenGLWidgets
- CMake 3.16+
- Visual Studio 2019/2022 or MinGW
- Boost 1.70+
- GLEW
- OpenGL drivers

Optional:
- CGAL
- FFTW
- GSL
- CUDA Toolkit 11+
- NVIDIA Cg Toolkit
```

#### Linux (Ubuntu/Debian)
```bash
# Required
sudo apt-get install \
    qt6-base-dev \
    qt6-base-dev-tools \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    cmake \
    build-essential \
    libboost-all-dev \
    libglew-dev \
    freeglut3-dev

# Optional
sudo apt-get install \
    libcgal-dev \
    libfftw3-dev \
    libgsl-dev \
    nvidia-cuda-toolkit
```

#### Linux (Fedora/RHEL)
```bash
# Required
sudo dnf install \
    qt6-qtbase-devel \
    qt6-qttools-devel \
    cmake \
    gcc-c++ \
    boost-devel \
    glew-devel \
    freeglut-devel

# Optional
sudo dnf install \
    CGAL-devel \
    fftw-devel \
    gsl-devel \
    cuda
```

### 6.2 Runtime Dependencies

#### Windows Binary Distribution
- Qt6 DLLs (Core, Gui, Widgets, Xml, OpenGL, OpenGLWidgets)
- Qt6 platform plugins (qwindows.dll)
- Visual C++ Runtime
- OpenGL32.dll (system)
- Boost DLLs (if dynamically linked)
- GLEW DLL
- Cg runtime (if using Cg shaders)

#### Linux Binary Distribution
- Qt6 libraries
- OpenGL libraries (Mesa or proprietary)
- X11 libraries
- Boost libraries
- Standard C++ library

---

## 7. Testing Checklist

### 7.1 Compilation Tests
- [ ] Clean build on Windows (VS2022)
- [ ] Clean build on Windows (MinGW)
- [ ] Clean build on Ubuntu 22.04+
- [ ] Clean build on Fedora 38+
- [ ] All optional modules compile
- [ ] No Qt deprecation warnings

### 7.2 Functional Tests
- [ ] Application launches
- [ ] Main window displays correctly
- [ ] Menu system works
- [ ] All dialogs open and close
- [ ] Volume data loading (various formats)
- [ ] Volume rendering displays
- [ ] Geometry rendering works
- [ ] Color table manipulation
- [ ] Isosurface extraction
- [ ] File save operations
- [ ] Import/Export functionality

### 7.3 Platform-Specific Tests

#### Windows
- [ ] High DPI scaling
- [ ] Multiple monitor support
- [ ] Window persistence
- [ ] File associations
- [ ] Installer runs correctly
- [ ] Uninstaller works

#### Linux
- [ ] Wayland compatibility
- [ ] X11 compatibility
- [ ] Desktop file integration
- [ ] File manager integration
- [ ] Package installation
- [ ] Dependency resolution

### 7.4 Optional Module Tests
- [ ] CGAL-dependent features (if enabled)
- [ ] CUDA acceleration (if enabled)
- [ ] Tiling operations
- [ ] Segmentation algorithms
- [ ] Advanced geometry operations

---

## 8. Risk Assessment and Mitigation

### High Risk Items

**1. QGLViewer Migration**
- **Risk**: Core 3D viewing may break
- **Mitigation**: Create isolated test case first, consider QGLViewer upstream updates
- **Fallback**: Implement minimal QOpenGLWidget wrapper

**2. OpenGL Context Management**
- **Risk**: Context creation failures on some systems
- **Mitigation**: Extensive testing on various GPUs, fallback to software rendering
- **Fallback**: Provide OpenGL configuration dialog

**3. CGAL Compatibility**
- **Risk**: CGAL may have issues with Qt6 types
- **Mitigation**: Test CGAL integration early, isolate Qt types from CGAL code
- **Fallback**: Keep modules optional, document known issues

### Medium Risk Items

**1. UI Form Compatibility**
- **Risk**: Some .ui forms may not convert cleanly
- **Mitigation**: Manual inspection in Qt Designer, test all dialogs
- **Fallback**: Recreate problematic forms from scratch

**2. Threading and Signals**
- **Risk**: Qt6 has stricter thread affinity rules
- **Mitigation**: Audit all cross-thread signal/slot connections
- **Fallback**: Use Qt::QueuedConnection explicitly

**3. Binary Distribution Size**
- **Risk**: Qt6 packages may be larger than Qt4
- **Mitigation**: Strip debug symbols, use windeployqt/linuxdeployqt
- **Fallback**: Document minimum installation, offer "lite" build

### Low Risk Items

**1. Performance Regression**
- **Risk**: Qt6 may be slower in some operations
- **Mitigation**: Profile critical paths, optimize if needed
- **Fallback**: Document performance characteristics

---

## 9. Timeline Estimate

**Total Duration**: 16-24 weeks (4-6 months)

- Phase 1 (Preparation): 1-2 weeks
- Phase 2 (CMake): 2-3 weeks  
- Phase 3 (Source): 4-6 weeks
- Phase 4 (Core): 3-4 weeks
- Phase 5 (Optional): 2-4 weeks
- Phase 6 (Testing): 2-3 weeks
- Phase 7 (Distribution): 2-3 weeks

Add 25% buffer for unexpected issues: **20-30 weeks (5-7.5 months)**

---

## 10. Success Criteria

✅ Application builds cleanly with Qt6 on Windows and Linux  
✅ No Qt deprecation warnings  
✅ All core functionality works identically to Qt4 version  
✅ 3D rendering performance within 10% of Qt4 version  
✅ All dialogs display and function correctly  
✅ Binary installers available for Windows and Linux  
✅ Documentation updated  
✅ No regressions in scientific accuracy  

---

## 11. Resources and References

### Qt6 Migration Guides
- Official Qt6 Porting Guide: https://doc.qt.io/qt-6/portingguide.html
- Qt6 CMake Manual: https://doc.qt.io/qt-6/cmake-manual.html
- Porting from Qt5 to Qt6: https://doc.qt.io/qt-6/porting-to-qt6-using-clazy.html

### Dependency Documentation
- CGAL Manual: https://doc.cgal.org/
- Boost Documentation: https://www.boost.org/doc/
- OpenGL Wiki: https://www.khronos.org/opengl/wiki/

### Build Tools
- CMake Documentation: https://cmake.org/documentation/
- CPack Guide: https://cmake.org/cmake/help/latest/module/CPack.html
- windeployqt: https://doc.qt.io/qt-6/windows-deployment.html
- linuxdeployqt: https://github.com/probonopd/linuxdeployqt

---

## 11. Migration Progress & Completed Changes

### Dependency Changes (December 2025)

#### GLEW: Bundled → System Dependency
**Completed**: December 8, 2025

**Rationale**: 
- The bundled GLEW version in `src/glew/` and `inc/glew/` was very old and unmaintained
- System package managers provide up-to-date, security-patched versions
- Reduces codebase maintenance burden
- Reduces binary distribution size

**Changes Made**:
1. Added `find_package(GLEW REQUIRED)` to main `CMakeLists.txt`
2. Removed `add_subdirectory(glew)` from `src/CMakeLists.txt`
3. Updated all references from `glew` target to `GLEW::GLEW` in:
   - `src/QGLViewer/CMakeLists.txt`
   - `src/VolumeRenderer/CMakeLists.txt`
   - `src/GeometryRenderer/CMakeLists.txt`
   - `src/SimpleExample/CMakeLists.txt`
4. Removed all `add_definitions(-DGLEW_STATIC)` flags from:
   - `src/VolumeRenderer/CMakeLists.txt`
   - `src/VolumeRover2/CMakeLists.txt`
5. Updated build documentation in `BUILD_QT6.md` and `INSTALL`

**Installation Requirements**:
- **Ubuntu/Debian**: `sudo apt-get install libglew-dev`
- **Fedora/RHEL**: `sudo dnf install glew-devel`
- **macOS**: `brew install glew`
- **Windows**: Download from http://glew.sourceforge.net/ or use vcpkg

**Legacy Code**: The old `src/glew/` and `inc/glew/` directories remain in the repository for reference but are no longer compiled. They may be removed in a future cleanup.

---

## 12. Next Steps

1. **Review this plan** with stakeholders
2. **Set up development environments** with Qt6
3. **Create migration branch** in git
4. **Start with Phase 1**: Environment setup and dependency verification
5. **Iterate through phases** systematically
6. **Regular progress reviews** (weekly recommended)
7. **Document issues** encountered and solutions

---

## Notes

- This is an archived project being brought back to life, so some investigation of current code state is needed
- Original Qt4 version may have been building on older OS versions; need to verify Windows 10/11 and modern Linux support
- Consider whether Qt6.5 LTS or latest Qt6.x is the better target
- Check if any commercial Qt modules were used (unlikely for academic project)
- The `zzzzzUNUSEDzzzzz` folder contains old code that should probably be ignored or removed
- VolumeRover2 appears to be the active application; VolumeRover 1.x code might be fully deprecated

---

**Document Version**: 1.0  
**Date**: December 8, 2025  
**Author**: GitHub Copilot  
**Status**: Draft - Ready for Review
