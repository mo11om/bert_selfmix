import os
import pytorch_lightning as pl

class GpuMonitoringCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("--- GPU Monitoring ---")
        print("--- NVIDIA SMI ---")
        os.system("nvidia-smi")
        print("--- AMD SMI ---")
        os.system("amd-smi")
        print("--------------------")
