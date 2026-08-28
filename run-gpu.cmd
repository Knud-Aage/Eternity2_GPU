@echo off
REM Runs the GPU solver from the repository root (pieces.csv is read by relative path).
REM
REM Prerequisites, once:
REM   powershell -File build-kernel.ps1     (produces SolveBlackwoodKernel.ptx)
REM   mvn clean package                     (produces target/classes and cp.txt below)
REM
REM cp.txt is the dependency classpath. Generate it with:
REM   mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

cd /d "%~dp0"

if not exist "SolveBlackwoodKernel.ptx" (
    echo ERROR: SolveBlackwoodKernel.ptx not found. Run: powershell -File build-kernel.ps1
    exit /b 1
)
if not exist "cp.txt" (
    echo ERROR: cp.txt not found. Run: mvn dependency:build-classpath -Dmdep.outputFile=cp.txt
    exit /b 1
)

for /f "usebackq delims=" %%A in ("cp.txt") do set CPFILE=%%A

REM 16384 measured best at production scale; see README.
if "%ETERNITY_GPU_NUM_THREADS%"=="" set ETERNITY_GPU_NUM_THREADS=16384

java -cp "target\classes;%CPFILE%" dk.puzzle.blackwood.BlackwoodGpuRunner
