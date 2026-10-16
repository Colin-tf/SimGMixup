import numpy as np
import os

import torch
class Logger(object):
    def __init__(self, args):
        self.test_results = [0 for _ in range(args.runs)]  
        self.val_results = [0 for _ in range(args.runs)]  
        self.train_results = [0 for _ in range(args.runs)]      
        self.args = args
        self.folder_name = f"./results{args.exp_tag}/{args.dataset}/{args.model}"

    def add_result(self, run, result, res_type="train"):
        assert 0 <= run < len(self.test_results)
        if res_type == "train":
            self.train_results[run] = result
        elif res_type == "val":
            self.val_results[run] = result
        elif res_type == "test":
            self.test_results[run] = result

    def dump_parameters(self, content):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        if self.args.mixup != "none":
            filename = os.path.join(self.folder_name, f"{self.args.model}_input_{self.args.edge_mixer}_{self.args.feat_mixer}_{self.args.pairing_mode}_{self.args.mixing_ratio}.txt")
        else:
            filename = os.path.join(self.folder_name, f"{self.args.model}.txt")
        with open(filename, "a+") as f:
            f.write(content + "\t")
            f.write(str(self.args))
            f.write("\n")

    def print_statistics(self):
        test_acc_mean = np.mean(self.test_results)
        test_acc_std = np.std(self.test_results)
        val_acc_mean = np.mean(self.val_results)
        val_acc_std = np.std(self.val_results)
        train_acc_mean = np.mean(self.train_results)
        train_acc_std = np.std(self.train_results)

        content = f'Test: {test_acc_mean:.2f} ± {test_acc_std:.2f} | Val: {val_acc_mean:.2f} ± {val_acc_std:.2f} | Train: {train_acc_mean:.2f} ± {train_acc_std:.2f}'
        print(content)
        self.dump_parameters(content)

    def save(self, dst_foler, dst_file, obj):
        if not os.path.exists(dst_foler):
            os.makedirs(dst_foler)
        file_extension = os.path.splitext(dst_file)[1]
        if file_extension == ".npy":
            np.save(os.path.join(dst_foler, dst_file), obj)
        elif file_extension == ".pt":
            torch.save(obj, os.path.join(dst_foler, dst_file))