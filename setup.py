"""
Builds csrc/ into genesis_l0/_l0native*.so. The library is loaded with ctypes
(see src/genesis_l0/_c_core.py), not imported as a Python module.

GENESIS_L0_NATIVE=1  -> -march=native (fastest, binary not portable)
default x86_64       -> -march=x86-64-v3 (AVX2 + POPCNT, Haswell / Zen 1 and newer)
"""
import os
import platform
import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    sys.exit("genesis-l0 supports Linux and macOS only (POSIX mlock/pthread). Use WSL on Windows.")

machine = platform.machine().lower()
# Python's sysconfig CFLAGS carry -fno-strict-overflow (= -fwrapv), which blocks
# vectorization of srp_dot and doubles query latency (measured 517 -> 281 us).
cflags = ["-O3", "-std=c99", "-Wall", "-Wextra", "-fPIC", "-fstrict-overflow"]

if os.environ.get("GENESIS_L0_NATIVE") == "1":
    cflags.append("-march=native")
elif machine in ("x86_64", "amd64"):
    cflags.append("-march=x86-64-v3")
elif sys.platform == "darwin" and machine == "arm64":
    cflags.append("-mcpu=apple-m1")
# Linux aarch64: baseline ARMv8 already has NEON `cnt` for popcount.

setup(
    ext_modules=[
        Extension(
            name="genesis_l0._l0native",
            sources=["csrc/l0_api.c", "csrc/srp_lsh.c"],
            depends=["csrc/l0_api.h", "csrc/srp_lsh.h"],
            include_dirs=["csrc"],
            define_macros=[("SRP_BITS", "256")],
            extra_compile_args=cflags,
            libraries=["m", "pthread"],
        )
    ],
)
