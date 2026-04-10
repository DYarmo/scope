# Scope

A Python-based camera control application for Thorlabs cameras, featuring GUI for image acquisition and analysis.

## Installation

### Prerequisites

- Python 3.8 or higher
- Thorlabs Camera SDK

### Thorlabs SDK Installation

The Thorlabs Scientific Imaging SDK is required for camera communication. Download and install it from the official Thorlabs website:

1. Visit [Thorlabs Scientific Imaging SDK Downloads](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam)
2. Download the appropriate SDK for your operating system (Windows/Linux)
3. Follow the installation instructions provided by Thorlabs
4. Ensure the SDK libraries are in your system's library path

#### Alternative: Using the SDK Package

If you have the SDK package file (`thorlabs_tsi_camera_python_sdk_package.zip`), you can install it using pip:

```bash
pip install thorlabs_tsi_camera_python_sdk_package.zip
```

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install PyQt5 pyqtgraph matplotlib numpy opencv-python pyserial
```

## Usage

Run the main application:

```bash
python PainSnap.py
```

## Features

- Camera control and image acquisition
- Real-time image display
- Image analysis tools
- Serial device integration
- GUI-based interface

## Project Structure

- `PainSnap.py` - Main GUI application
- `PainSnap_tools.py` - Camera control utilities
- `camfind.py` - Camera discovery
- `image_analysis.py` - Image processing functions
- `tcam.py` - Thorlabs camera interface
- `main.py` - Alternative entry point
- `palettes/` - Color palettes for image display