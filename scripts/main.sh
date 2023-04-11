#!/bin/bash
#export CARLA_ROOT=/home/quan/carla_0.9.10
#export CHALLENGE_DIR=/home/quan/Cadre                          # change to where you clone this project
export CARLA_ROOT=/home/liuchi/zhaoyinuo/carla_0.9.10
export CHALLENGE_DIR=/home/liuchi/zhaoyinuo/Cadre                          # change to where you clone this project

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg           # 0.9.10
export PYTHONPATH=$PYTHONPATH:$CHALLENGE_DIR/leaderboard
export PYTHONPATH=$PYTHONPATH:$CHALLENGE_DIR/scenario_runner
export HAS_DISPLAY='0'

python main.py