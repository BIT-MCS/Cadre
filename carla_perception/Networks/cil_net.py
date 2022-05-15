import torch
import torch.nn as nn
# from torch.nn import functional as F


class CarlaNet(nn.Module):
    def __init__(self, dropout_vec=None):
        super(CarlaNet, self).__init__()
        self.conv_block = nn.Sequential(
            # [c, 200, 88], [c, 256, 144]
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            # [c, 98, 42], [c, 126, 70]
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[0]),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            # [c, 96, 40], [c, 124, 68]
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[1]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            # [c, 47, 19], [c, 61, 33]
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # [c, 45, 17], [c, 59, 31]
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[3]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            # [c, 22, 8], [c, 29, 15]
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[4]),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            # [c, 20, 6], [c, 27, 13]
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[5]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            # [c, 18, 4], [c, 25, 11]
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[6]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            # [c, 16, 2], [c, 23, 9]
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[7]),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
                nn.Linear(52992, 512),
                # nn.Linear(8192, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.control_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, img, speed):
        img = self.conv_block(img)
        img = img.view(-1, 52992)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        pred_control = torch.cat(
            [out(emb) for out in self.control_branches], dim=1)
        pred_speed = self.speed_branch(img)
        return pred_control, pred_speed, img, emb


class UncertainNet(nn.Module):
    def __init__(self, structure=2, dropout_vec=None):
        super(UncertainNet, self).__init__()
        self.structure = structure

        if (self.structure < 2 or self.structure > 3):
            raise("Structure must be one of 2|3")

        self.uncert_speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        if self.structure == 2:
            self.uncert_control_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),
                ) for i in range(4)
            ])

        if self.structure == 3:
            self.uncert_control_branches = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),
            )

    def forward(self, img_emb, emb):
        if self.structure == 2:
            log_var_control = torch.cat(
                [un(emb) for un in self.uncert_control_branches], dim=1)
        if self.structure == 3:
            log_var_control = self.uncert_control_branches(emb)
            log_var_control = torch.cat([log_var_control for _ in range(4)],
                                        dim=1)

        log_var_speed = self.uncert_speed_branch(img_emb)

        return log_var_control, log_var_speed


class CilFinalNet(nn.Module):
    def __init__(self, net_params):
        super(CilFinalNet, self).__init__()
        self.structure = net_params['structure']
        self.dropout_vec = net_params['dropout_vec']

        self.carla_net = CarlaNet(dropout_vec=self.dropout_vec)
        self.uncertain_net = UncertainNet(self.structure)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(
        #             m.weight)
        #         m.bias.data.fill_(0.01)

    def forward(self, img, speed):
        pred_control, pred_speed, img_emb, emb = self.carla_net(img, speed)
        log_var_control, log_var_speed = self.uncertain_net(img_emb, emb)

        return pred_control, pred_speed, log_var_control, log_var_speed

def get_model(net_params):
    return CilFinalNet(net_params)