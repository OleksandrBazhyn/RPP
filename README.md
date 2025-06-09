# Parallel String Hashing for Fast Large-Scale Text Comparison

## Overview

This project demonstrates and compares three implementations of a fast string matching algorithm for large text files, based on **polynomial string hashing** (Rabin-Karp).  
The three versions are:

- **Simple C++** — Single-threaded, standard C++
- **OpenMP** — Multithreaded, for shared-memory systems
- **MPI** — Multiprocess, for distributed-memory clusters

The purpose is to compare performance, scalability, and programming models for the classic problem of efficiently finding matching lines between two large text files.
