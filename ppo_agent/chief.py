import torch.nn as nn
import time
import datetime
import os
import csv


def chief(update_threshold, traffic_light, counter, shared_model_list, shared_grad_buffers, optimizer,
          son_process_counter, max_grad_norm, total_thread):
    while True:
        time.sleep(1)
        if counter.get() >= update_threshold:
            optimizer.zero_grad()
            for shared_model_name in shared_model_list:
                shared_model = shared_model_list[shared_model_name]
                for n, p in shared_model.named_parameters():
                    if p.requires_grad:
                        p._grad = shared_grad_buffers.grads[shared_model_name + '_' + n + '_grad'].clone().detach()
                nn.utils.clip_grad_norm_(shared_model.parameters(), max_grad_norm)

            optimizer.step()
            shared_grad_buffers.reset()
            counter.reset()
            traffic_light.switch()  # workers start new loss computation
        if son_process_counter.get() >= total_thread:
            print('chief finished.')
            break
