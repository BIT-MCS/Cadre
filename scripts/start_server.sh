#!/bin/bash
export CARLA_ROOT=/home/quan/carla_0.9.10 # [PATH TO YOUR LOCAL DIRECTORY WITH CaraUE4.sh]

SDL_HINT_CUDA_DEVICE=1 ${CARLA_ROOT}/CarlaUE4.sh -carla-world-port=8010 -quality-level=Epic -resx=800 -resy=600 -opengl
#sleep 5
#SDL_HINT_CUDA_DEVICE=1 ${CARLA_ROOT}/CarlaUE4.sh -carla-world-port=8020 -quality-level=Epic -resx=800 -resy=600 -opengl&