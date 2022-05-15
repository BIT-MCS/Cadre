#!/bin/bash
#SDL_HINT_CUDA_DEVICE=0 /home/quan/carla_0.9.10/CarlaUE4.sh -carla-world-port=8010 -quality-level=Epic -resx=800 -resy=600 -opengl &
##sleep 10
#SDL_HINT_CUDA_DEVICE=0 /home/quan/carla_0.9.10/CarlaUE4.sh -carla-world-port=8020 -quality-level=Epic -resx=800 -resy=600 -opengl
##sleep 10
SDL_HINT_CUDA_DEVICE=0 /home/liuchi/zhaoyinuo/nips/carla_0.9.10/CarlaUE4.sh -carla-world-port=8010 -quality-level=Epic -resx=800 -resy=600 -opengl &
#sleep 10
SDL_HINT_CUDA_DEVICE=0 /home/liuchi/zhaoyinuo/nips/carla_0.9.10/CarlaUE4.sh -carla-world-port=8020 -quality-level=Epic -resx=800 -resy=600 -opengl
#sleep 10