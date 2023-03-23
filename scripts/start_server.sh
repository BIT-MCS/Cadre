#!/bin/bash
export CARLA_ROOT=/home/quan/carla_0.9.10

SDL_HINT_CUDA_DEVICE=0 ${CARLA_ROOT}/CarlaUE4.sh -carla-world-port=8010 -quality-level=Epic -resx=800 -resy=600 -opengl