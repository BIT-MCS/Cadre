from torch.nn import CrossEntropyLoss, BCELoss
import os
from Config.auto_basic_config import basic_config
import csv

class danet_config(basic_config):
    def __init__(self):
        basic_config.__init__(self)

        self.local_rank = -1

        self.phase = 'train' #'test' or 'train'
        self.load_epoch = 48 #-1

        # 1: 1 center camera
        # 2: 4 center camera
        # 3: 1 center camera, lidar
        # 4: 4 center camera, lidar
        # 5: 1 center camera, route
        # 6: 1 center camera, 1 left camera, 1 right camera, route
        # 7: 1 center camera, route, speed (as img)
        # 8: 1 center camera, route, lidar, speed (as img)
        # 9: 1 center camera, route, speed (only for bc)
        self.input_mode = 9 #3
        self.change_input_mode(self.input_mode)

        # 0: last camera
        # 1: last camera, light status, light dist
        # 2: last camera, topdown rgb, light status, light dist
        # 3: last camera, topdown semantic, light status, light dist
        # 4: last camera, lidar, light status, light dist
        # 5: last camera, lidar, topdown rgb, light status, light dist
        # 6: last camera, lidar, topdown semantic, light status, light dist
        # 7: last camera semantic
        # 8: last camera semantic, route fig
        # 9: last camera semantic, route fig, light state
        # 10: center, left, right camera semantic, route fig
        # 11: center, left, right camera semantic, route fig, light state
        # 12: last camera semantic, route fig, light state, behviour cloning
        self.output_mode = 12 #8 #5
        self.change_output_mode(self.output_mode)

        self.version = 'danet'
        # self.train_town = "town01" # "all"
        # self.train_town = "nocrash01" # "all"
        # self.train_town = "nocrashBa" # "all"
        # self.train_town = "nocrashBaNew" # "all"
        # self.train_town = "challenge75" # "all"
        # self.train_town = "challenge_n10_k123_r5" # "all"
        # self.train_town = 'nocrash_IL_n10_k1234_r40'
        self.train_town = 'challenge_IL_n10_k1234_r40'

        version_suffix = str(self.input_mode) + str(self.output_mode)

        self.exp_suffix = '36'
        self.exp_dir = self.version + version_suffix + "_" + self.train_town

        
        ############ train opt ################
        self.train_opt = dict()
        self.train_opt['data_name'] = 'carla_percept'
        self.train_opt['route_index'] = 'invaild' #self.town_route_dict[self.train_town]
        self.train_opt['batch_size'] = 16 #48 #32
        self.train_opt['num_workers'] = 4
        self.train_opt['in_num_frames'] = self.in_backbone
        self.train_opt['phase'] = 'train'
        self.train_opt['input_mode'] = self.input_mode
        self.train_opt['output_mode'] = self.output_mode
        # self.train_opt['root_dir'] = '/data2/wk/datasets/3camera_seg10/train'
        # self.train_opt['root_dir'] = '/data2/wk/datasets/vae_data_noise10_actor55_rep3/train'
        # self.train_opt['root_dir'] = '/data2/wk/datasets/vae_nocrash_n10_c20_p50_r4/train'
        # self.train_opt['root_dir'] = '/data2/wk/datasets/vae_challenge_n10_c55_p50_r6/train'
        # self.train_opt['root_dir'] = '/data2/wk/datasets/vae_nocrash_balanced/train'
        # self.train_opt['root_dir'] = '/data2/wk/datasets/vae_nocrash_balanced_new/train'
        # self.train_opt['root_dir'] = '/data2/wk/datasets/vae_challenge_75routes/train'
        self.train_opt['root_dir'] = '/data2/wk/datasets/challenge_IL_n10_k1234_r40/train'
        self.train_opt['img_transform_params'] = self.img_transform_optionsF
        self.train_opt['topdown_transform_params'] = self.topdown_rgb_transform_optionsF
        self.train_opt['lidar_transform_params'] = self.lidar_transform_optionsF

        ############ test opt ################
        self.test_opt = dict()
        self.test_opt['data_name'] = 'carla_percept_test'
        self.test_opt['route_index'] = [21, 22, 23, 24] #[9] #[1]
        self.test_opt['batch_size'] = 1
        self.test_opt['num_workers'] = 0
        self.test_opt['in_num_frames'] = self.in_backbone
        self.test_opt['phase'] = 'test'
        self.test_opt['input_mode'] = self.input_mode
        self.test_opt['output_mode'] = self.output_mode
        # self.test_opt['root_dir'] = '/data2/wk/datasets/3camera_seg10/test'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/3camera_seg10/train'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/vae_data_noise10_actor55_rep3/test'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/vae_nocrash_n10_c20_p50_r4/train'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/vae_challenge_n10_c55_p50_r6/train'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/vae_nocrash_balanced/train'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/vae_nocrash_balanced_new/train'
        # self.test_opt['root_dir'] = '/data2/wk/datasets/vae_challenge_75routes/train'
        self.test_opt['root_dir'] = '/data2/wk/datasets/vae_IL_nocrash_n10_k1234_r40_test/train'
        self.test_opt['route_name'] = 'route_' + str(self.test_opt['route_index'][0])[:-1]
        self.test_opt['img_transform_params'] = self.img_transform_optionsF
        self.test_opt['topdown_transform_params'] = self.topdown_rgb_transform_optionsF
        self.test_opt['lidar_transform_params'] = self.lidar_transform_optionsF

        self.networks = dict()
        self.optimizers = dict()

        ###########  AutoEncoder ################
        self.networks['autoencoder'] = dict()
        self.networks['autoencoder']['net_name'] = 'autoencoder'
        self.networks['autoencoder']['model_name'] = 'danet'
        self.networks['autoencoder']['input_channel'] = self.in_backbone * 3
        if self.in_left_camera:
            self.networks['autoencoder']['input_channel'] += self.in_backbone * 3
        if self.in_right_camera:
            self.networks['autoencoder']['input_channel'] += self.in_backbone * 3
        if self.in_lidar:
            self.networks['autoencoder']['input_channel'] += self.in_backbone * 3
        if self.in_route:
            self.networks['autoencoder']['input_channel'] += self.in_backbone * 1
        if self.in_speed:
            self.networks['autoencoder']['input_channel'] += 1

        if self.pred_camera_seg:
            # 0.9.10: 8 classes: unlabeled, road, car, person, 
            # (build, wall), (fence, Pole, TrafficSign, Static, Dynamic)
            # (Vegetation, Water, Terrain), Road line
            # 0.9.9.4" 8 classes: unlabeled, (road), car, person, 
            # (build, wall), (fence, Pole)
            # (Vegetation), Road line
            self.networks['autoencoder']['camera_output_channel'] = 8
        else:
            self.networks['autoencoder']['camera_output_channel'] = 3
        
        if self.pred_left_camera_seg:
            self.networks['autoencoder']['left_camera_output_channel'] = 8
        else:
            self.networks['autoencoder']['left_camera_output_channel'] = 3
        
        if self.pred_right_camera_seg:
            self.networks['autoencoder']['right_camera_output_channel'] = 8
        else:
            self.networks['autoencoder']['right_camera_output_channel'] = 3

        self.networks['autoencoder']['light_classes_num'] = 4
        self.networks['autoencoder']['z_dims'] = 256 #64
        #['transformer', 'position'] only when self.pred_bc is True
        self.networks['autoencoder']['att_type'] = 'transformer'
        self.networks['autoencoder']['da_feature_channel'] = 512 #128
        self.networks['autoencoder']['inter_att_dims'] = 512 #128
        self.networks['autoencoder']['pred_light_state'] = self.pred_light_state
        self.networks['autoencoder']['pred_light_dist'] = self.pred_light_dist
        self.networks['autoencoder']['pred_lidar'] = self.pred_lidar
        self.networks['autoencoder']['pred_topdown_rgb'] = self.pred_topdown_rgb
        self.networks['autoencoder']['pred_topdown_seg'] = self.pred_topdown_seg
        self.networks['autoencoder']['pred_route'] = self.pred_route
        self.networks['autoencoder']['pred_camera_seg'] = self.pred_camera_seg
        self.networks['autoencoder']['pred_left_camera_seg'] = self.pred_left_camera_seg
        self.networks['autoencoder']['pred_right_camera_seg'] = self.pred_right_camera_seg
        self.networks['autoencoder']['pred_bc'] = self.pred_bc
        self.networks['autoencoder']['in_bc_speed'] = self.in_bc_speed
        if self.load_epoch == -1:
            self.networks['autoencoder']['pretrained_path'] = None
            self.networks['autoencoder']['pretrained'] = False
        else:
            self.networks['autoencoder']['pretrained'] = True
            file_name = 'net_epoch' + str(self.load_epoch)
            self.networks['autoencoder']['pretrained_path'] = os.path.join('carla_perception/Experiments'+self.exp_suffix,
                                                                self.exp_dir,
                                                                file_name)

        self.optimizers['autoencoder'] = dict()
        self.optimizers['autoencoder']['type'] = 'adam'
        self.optimizers['autoencoder']['lr'] = 0.0001
        self.optimizers['autoencoder']['beta'] = (0.9, 0.999)#(0.5, 0.999)
        self.optimizers['autoencoder']['weight_decay'] = 5e-4
        self.optimizers['autoencoder']['lr_scheduler'] = 'CosineLR'
        self.optimizers['autoencoder']['t_max'] = self.max_num_epochs
        if self.load_epoch == -1:
            self.optimizers['autoencoder']['pretrained_path'] = None
        else:
            file_name = 'optim_epoch' + str(self.load_epoch)
            self.optimizers['autoencoder']['pretrained_path'] = os.path.join('carla_perception/Experiments'+self.exp_suffix,
                                                                    self.exp_dir,
                                                                    file_name)
        
        self.metric1 = 'L1Loss'
        self.criterions = dict()

    def log_info(self, path):
        try:
            log_file = open(path, 'a', newline='')
        except IOError:
            print(path, ' has not been created!')
            return
        file_writer = csv.writer(log_file)
        file_writer.writerow(['config_ppo_model'])
        for p in self.__dict__:
            if p[0] == '_':
                continue
            file_writer.writerow([p, str(self.__getattribute__(p))])
        log_file.close()

config = danet_config()



