@echo off
rem duckie.cpp build: RTX 40-series (sm_89) only, CUDA + native CPU, static exes.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build\build.ninja (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real ^
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" ^
    -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF || exit /b 1
)
cmake --build build -j 10 --target llama-server llama-bench llama-cli test-backend-ops %* || exit /b 1
