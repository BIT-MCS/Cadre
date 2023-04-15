THROTTLE_CONTROL = {
    0: [0, 0],
    1: [0, 1],
    2: [0.6, 0]
}

STEER_CONTROL = {
    0: -8. / 16., 1: -7. / 16., 2: -6. / 16., 3: -5. / 16., 4: -4. / 16.,
    5: -3. / 16., 6: -2. / 16., 7: -1. / 16., 8: 0.0, 9: 1. / 16.,
    10: 2. / 16., 11: 3. / 16., 12: 4. / 16., 13: 5. / 16., 14: 6. / 16.,
    15: 7. / 16., 16: 8. / 16., 17: 9. / 16, 18: -9. / 16, 19: 10. / 16,
    20: -10. / 16, 21: 11. / 16, 22: -11. / 16, 23: 12. / 16., 24: -12. / 16.,
    25: 13. / 16, 26: -13. / 16., 27: 14. / 16., 28: -14. / 16., 29: 15. / 16.,
    30: -15. / 16., 31: 1., 32: -1.
}

rollout_cfg = dict(
    num_steps=200,
    mini_batch_num=2,
    feature_dims=512 + 18,
    seq_length=8,
    use_gae=True,
    gamma=0.99,
    tau=0.95
)


agent_cfg = dict(
    rank=-1,
    model_cfg=dict(
        use_lstm=True,
        vae_device=0,
        device_num=0,
        vae_params="CoPM",
        # vae_params="CoPM w/o att",
        measurement_dim=18,
        num_output=dict(steer=len(STEER_CONTROL), throttle=len(THROTTLE_CONTROL)),
        command_num=4,

    ),
    frame=8,
    STEER_CONTROL=STEER_CONTROL,
    THROTTLE_CONTROL=THROTTLE_CONTROL,
    ent_coeff=0.01,
    value_coeff=0.1,
    clip_coeff=1.,
    clip=0.1,

)

eval_cfg = dict(
    # !!! config your model path here
    pretrained_path='result/04-15/18-08-46/0/',
    # !!! config the explicit episode model you want to use for evaluation, recommend 8 models for safety
    load_episode=['2900', '2800', '2700', '2600', '2500', '2400'],
    eval_episode=25,
)


env_cfg = dict(
    root_path="result",
    pretrained_path="",
    debug=0,
    frame_rate=10,
    timeout=60,
    client_timeout=60,
    vehicle_block_time=400,
    min_speed=5,
    max_speed=9,
    target_speed=7,
    max_degree=90,
    host="localhost",
    # todo: change to False when eval
    training=False,
    route_indexer="priority",
    num_processes=1,
    port=[8010],
    town=["Town01"],
    # amount: [vehicle numbers, pedestrian numbers]
    amount=[20, 50],

    routes=['nocrash_route/eval_routes/Nocrash_Town01.xml'],
    # routes=['nocrash_route/Nocrash_Town02.xml'],
    scenarios=['nocrash_scenarios/empty_scenario.json'],
    sensor_list=[
        {
            'type': 'sensor.camera.rgb',
            'x': 1.3, 'y': 0.0, 'z': 1.3,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 256, 'height': 144, 'fov': 90,
            'id': 'rgb'
        },
        {
            'type': 'sensor.other.imu',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'sensor_tick': 0.05,
            'id': 'imu'
        },
        {
            'type': 'sensor.other.gnss',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'sensor_tick': 0.01,
            'id': 'gps'
        },
        {
            'type': 'sensor.speedometer',
            'reading_frequency': 20,
            'id': 'speed'
        },
        {
            'type': 'sensor.other.obstacle',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'id': 'obstacle'
        },
    ],
)
