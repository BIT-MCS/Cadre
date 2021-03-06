# Cadre
This is the code accompanying the paper: "CADRE: A Cascade Deep Reinforcement Learning Framework for Vision-based Autonomous Urban Driving" by Yinuo Zhao, Kun Wu, et. al, published at AAAI 2022.

## :page_facing_up: Description
CADRE is a novel CAscade Deep REinforcement learning framework to achieve model-free vision-based autonomous urban driving on CARLA benchmark. We also provide an environment wrapper for CARLA that is suitable for distributed DRL training.

### Installation
1. Clone repo
    ```
    git clone https://github.com/BIT-MCS/Cadre.git
    cd Cadre
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
3. Download the trained perception model from [Google Driver](https://drive.google.com/drive/folders/1W00ZJ_807QcSgbQrEgCmiQznIbdlnVjX?usp=sharing) under `carla_perception/`

4. Download the Carla 0.9.10 server from [official website](https://carla.readthedocs.io/en/0.9.10/start_quickstart/). 
## :computer: Training

To start server, run the script under `scripts/start_server.sh`. Make sure to replace the `CARLA_ROOT` to your own directory.

To start training client, we provide a training script under `scripts/main.sh`, you need to change the `CARLA_ROOT` and `CHALLENGE_DIR`.

```
#!/bin/bash
export CARLA_ROOT=[PATH TO YOUR LOCAL DIRECTORY WITH CaraUE4.sh]
export CHALLENGE_DIR=[PATH TO WHERE]

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg           # 0.9.10
export PYTHONPATH=$PYTHONPATH:$CHALLENGE_DIR/leaderboard
export PYTHONPATH=$PYTHONPATH:$CHALLENGE_DIR/scenario_runner
export HAS_DISPLAY='0'

python main.py
```
## :scroll: Acknowledgement
This work was supported in part by Shanghai Pujiang Program and the National Research and Development Program of China (No.
2019YQ1700).

## :e-mail: Contact

If you have any question, please email `ynzhao@bit.edu.cn`.

## Paper
If you are interested in our work, please cite our paper as

```
@inproceedings{zhao2022cadre,
  author    = {Zhao, Yinuo and Wu, Kun and Xu, Zhiyuan and Che, Zhengping and Lu, Qi and Tang, Jian and Liu, Chi Harold},
  title     = {CADRE: A Cascade Deep Reinforcement Learning Framework for Vision-based Autonomous Urban Driving},
  booktitle = {Association for the Advancement of Artificial Intelligence (AAAI)},
  year      = {2022},
}
```
