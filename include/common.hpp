// Copyright 2026 depth-anything-v2.cpp contributors
// Licensed under the Apache License, Version 2.0

#pragma once
#include <cstdio>
#define DA_LOG(...)  do { std::fprintf(stderr, "[depth-anything] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
