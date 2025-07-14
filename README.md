# MPC_CPP - Model Predictive Control C++ Implementation (3DoF-KF MPC)

A C++ implementation of Three-Degree-of-Freedom Kalman Filter Model Predictive Control (MPC) for a SIR (Susceptible-Infected-Recovered) epidemic model using OSQP solver and Eigen library. 
See Banerjee et al. 2025 for algorithm details. https://doi.org/10.1021/acs.iecr.4c04583

![kalman_schem_automatica](https://github.com/user-attachments/assets/43e342b0-42b6-4035-83f6-31b6b4365d5c)

## Features

- **MPC Controller**: Implements a 3DoF-KF Model Predictive Controller with prediction and control horizons
- **SIR Model**: Highly Nonlinear Epidemic model. Set of Differential Equations. Used to benchmark the control algorithm.
- **OSQP Solver**: Uses OSQP (Operator Splitting Quadratic Program) for optimization
- **Eigen Integration**: Leverages Eigen library for matrix operations
- **Three-DOF Tuning**: Implements three-degree-of-freedom controller tuning
- **Constraint Handling**: Supports input, output, and rate constraints

## Dependencies

### Required Libraries
- **Eigen 3.4.0+**: Linear algebra library
- **OSQP**: Operator Splitting Quadratic Program solver
- **OsqpEigen**: C++ wrapper for OSQP
- **Boost 1.88.0+**: C++ libraries (iostreams component)
- **CMake 3.14+**: Build system

### System Requirements
- **Compiler**: C++17 compatible compiler (MSVC, GCC, Clang)
- **OS**: Windows, Linux, macOS

## Installation

### Option 1: Using vcpkg (Recommended)

```bash
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # On Windows: .\bootstrap-vcpkg.bat

# Install dependencies
./vcpkg install eigen3
./vcpkg install osqp
./vcpkg install osqp-eigen
./vcpkg install boost-iostreams

# Integrate with your system
./vcpkg integrate install
```

### Option 2: Manual Installation

1. **Eigen**: Download from [eigen.tuxfamily.org](https://eigen.tuxfamily.org/)
2. **OSQP**: Follow instructions at [osqp.org](https://osqp.org/)
3. **OsqpEigen**: Clone from [GitHub](https://github.com/robotology/osqp-eigen)
4. **Boost**: Download from [boost.org](https://www.boost.org/)

**Note**: If using manual installation, you may need to set CMake variables to point to your library locations:
```bash
cmake .. -DEIGEN3_ROOT=/path/to/eigen -DOSQP_PREFIX_PATH=/path/to/osqp -DOSQPEIGEN_PREFIX_PATH=/path/to/osqp-eigen -DBOOST_ROOT=/path/to/boost
```

## Building the Project

### Using CMake

```bash
# Clone the repository
git clone <your-repo-url>
cd MPC_CPP

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Release
```

### Using Visual Studio

1. Open the project folder in Visual Studio
2. Select "CMake" as the project type
3. Build the project using Ctrl+Shift+B

## Project Structure

```
MPC_CPP/
├── src/
│   ├── controller.cpp      # MPC controller implementation
│   ├── MPC_main.cpp        # Main simulation loop
│   └── system_ode.cpp      # SIR model ODE solver
├── include/
│   ├── Header1.h           # Main header with classes and functions
│   ├── globals.h           # Global constants and parameters
│   └── gnuplot-iostream.h  # Plotting utilities
├── CMakeLists.txt          # CMake configuration
└── README.md              # This file
```

## Usage

### Running the Simulation

```bash
# After building
./MPC_CPP  # On Windows: MPC_CPP.exe
```

### Configuration

Key parameters can be modified in `include/globals.h`:

- **Simulation Settings**: `ts`, `Tstop`, `p`, `m`
- **Model Parameters**: `Betai`, `Gammai`, `kvi`, `mu`
- **Controller Tuning**: `tau_r`, `tau_d`, `tau_u`
- **Constraints**: `y_min`, `y_max`, `u_min`, `u_max`, `del_u_min`, `del_u_max`

## Controller Features

### MPC Formulation
- **Prediction Horizon**: `p = 10` time steps
- **Control Horizon**: `m = 8` time steps
- **Sampling Time**: `ts = 1` time unit

### State-Space Model
- **States**: 2 (Susceptible, Infected)
- **Inputs**: 1 (Control input)
- **Disturbances**: 1 (Unmeasured disturbance) 1 (Measured Disturbance)
- **Outputs**: 1 (Infected population)

### Constraints
- Input constraints (actuator limits)
- Output constraints (safety limits)
- Rate constraints (actuator dynamics)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Acknowledgments

- OSQP developers for the optimization solver
- Eigen developers for the linear algebra library
- Boost developers for the C++ libraries 

## Citation

If you use this code in your research, please cite:

**Banerjee, S., Khan, O., El Mistiri, M., Nandola, N. N., Hekler, E., & Rivera, D. E. (2025). Data-Driven Control of Nonlinear Process Systems Using a Three-Degree-of-Freedom Model-on-Demand Model Predictive Control Framework. *Industrial & Engineering Chemistry Research*, 64(17), 8847-8864.** https://doi.org/10.1021/acs.iecr.4c04583

### BibTeX
```bibtex
@article{banerjee2025INECR,
  author = {Banerjee, Sarasij and Khan, Owais and El Mistiri, Mohamed and Nandola, Naresh N. and Hekler, Eric and Rivera, Daniel E.},
  title = {Data-Driven Control of Nonlinear Process Systems Using a Three-Degree-of-Freedom Model-on-Demand Model Predictive Control Framework},
  journal = {Industrial \& Engineering Chemistry Research},
  volume = {64},
  number = {17},
  pages = {8847-8864},
  year = {2025},
  doi = {10.1021/acs.iecr.4c04583},
  url = {https://doi.org/10.1021/acs.iecr.4c04583}
}
```
