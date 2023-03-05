#!/bin/env python

"""
Extract approximating LoRA by SVD from two SD models
Based on: <https://github.com/kohya-ss/sd-scripts/blob/main/networks/train_network.py>

Train LoRA with custom preprocessing, tagging and bucketing

Disabled/broken:
- `accelerate` with *dynamo* enabled
- `xformers` due to *faketensors* requirement
- `mem_eff_attn` due to *forwardfunc* mismatch
- 'use_8bit_adam` due to *bitsandbyttes* CUDA errors
"""

import os
import gc
import sys
import json
import time
import argparse
import tempfile
import torch
import logging
import importlib
import transformers
from pathlib import Path
from modules.util import log, Map, get_memory
import modules.process
import modules.sdapi

latents = importlib.import_module('modules.lora-latents')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'lora'))
from train_network import train


options = Map({
    "v2": False,
    "v_parameterization": False,
    "pretrained_model_name_or_path": "",
    "train_data_dir": "",
    "shuffle_caption": False,
    "caption_extension": ".txt",
    "caption_extention": ".txt",
    "keep_tokens": None,
    "color_aug": False,
    "flip_aug": False,
    "face_crop_aug_range": None,
    "random_crop": False,
    "debug_dataset": False,
    "resolution": "512,512",
    "cache_latents": True,
    "enable_bucket": False,
    "min_bucket_reso": 256,
    "max_bucket_reso": 1024,
    "bucket_reso_steps": 64,
    "bucket_no_upscale": False,
    "reg_data_dir": None,
    "in_json": "",
    "dataset_repeats": 1,
    "output_dir": "",
    "output_name": "",
    "save_precision": "fp16",
    "save_every_n_epochs": None,
    "save_n_epoch_ratio": None,
    "save_last_n_epochs": None,
    "save_last_n_epochs_state": None,
    "save_state": False,
    "resume": None,
    "max_grad_norm": 0.0,
    "train_batch_size": 1,
    "max_token_length": None,
    "use_8bit_adam": False,
    "mem_eff_attn": False,
    "xformers": False,
    "vae": None,
    "learning_rate": 1e-04,
    "max_train_steps": 8000,
    "max_train_epochs": None,
    "max_data_loader_n_workers": 8,
    "persistent_data_loader_workers": False,
    "seed": 42,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "fp16",
    "full_fp16": False,
    "clip_skip": None,
    "logging_dir": None,
    "log_prefix": None,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 0,
    "prior_loss_weight": 1.0,
    "no_metadata": False,
    "save_model_as": "ckpt",
    "unet_lr": 0.001,
    "text_encoder_lr": 5e-05,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1,
    "network_weights": None,
    "network_module": "networks.lora",
    "network_dim": 16,
    "network_alpha": 1.0,
    "network_args": None,
    "network_train_unet_only": False,
    "network_train_text_encoder_only": False,
    "training_comment": "mood-magic",
    "caption_dropout_rate": 0.0,
    "caption_dropout_every_n_epochs": None,
    "caption_tag_dropout_rate": 0.0,
})


def mem_stats():
    gc.collect()
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    mem = get_memory()
    log.info({ 'memory': { 'ram': mem.ram, 'gpu': mem.gpu } })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train lora')
    parser.add_argument('--model', type=str, default=None, required=True, help='original model to use a base for training')
    parser.add_argument('--input', type=str, default=None, required=True, help='input folder with training images')
    parser.add_argument('--output', type=str, default=None, required=True, help='lora name')
    parser.add_argument('--tag', type=str, default=None, required=False, help='primary tag')
    parser.add_argument('--dir', type=str, default=None, required=True, help='folder containing lora checkpoints')
    parser.add_argument('--interim', type=int, default=0, help = 'save interim checkpoints after n epoch')
    parser.add_argument('--noprocess', default = False, action='store_true', help = 'skip processing and use existing input data')
    parser.add_argument('--notrain', default = False, action='store_true', help = 'just run processing and skip training')
    parser.add_argument('--nocaptions', default = False, action='store_true', help = 'skip creating captions and tags')
    parser.add_argument('--nolatents', default = False, action='store_true', help = 'skip generating vae latents')
    parser.add_argument('--offline', default = False, action='store_true', help = 'do not use webui server for processing')
    parser.add_argument('--shutdown', default = False, action='store_true', help = 'shutdown webui server')
    parser.add_argument('--gradient', type=int, default=1, required=False, help='gradient accumulation steps, default: %(default)s')
    parser.add_argument('--steps', type=int, default=5000, required=False, help='training steps, default: %(default)s')
    parser.add_argument('--dim', type=int, default=128, required=False, help='network dimension, default: %(default)s')
    parser.add_argument('--repeats', type=int, default=10, required=False, help='number of repeats per image, default: %(default)s')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size, default: %(default)s')
    parser.add_argument('--lr', type=float, default=1e-04, required=False, help='model learning rate, default: %(default)s')
    parser.add_argument('--unetlr', type=float, default=1e-04, required=False, help='unet learning rate, default: %(default)s')
    parser.add_argument('--textlr', type=float, default=5e-05, required=False, help='text encoder learning rate, default: %(default)s')
    parser.add_argument('--dreambooth', default=False, action='store_true', help = "use dreambooth style training")
    parser.add_argument('--debug', default=False, action='store_true', help = "enable debug logging")
    args = parser.parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        log.error({ 'lora cannot find model': args.model })
        exit(1)
    options.pretrained_model_name_or_path = args.model
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        log.error({ 'lora cannot find training dir': args.input })
        exit(1)
    if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
        log.error({ 'lora cannot find training dir': args.dir })
        exit(1)
    options.output_dir = args.dir
    options.output_name = args.output
    options.max_train_steps = args.steps
    options.network_dim = args.dim
    options.gradient_accumulation_steps = args.gradient
    options.save_every_n_epochs = args.interim if args.interim > 0 else None
    options.learning_rate = args.lr
    options.unet_lr = args.unetlr
    options.text_encoder_lr = args.textlr
    options.train_batch_size = args.batch
    log.info({ 'train lora args': vars(options) })
    transformers.logging.set_verbosity_error()
    mem_stats()

    concept = 'lora'
    if args.tag is not None:
        concept = args.tag.split(',')[0].strip()
    dir = os.path.join(tempfile.gettempdir(), args.output, str(args.repeats) + '_' + concept)
    Path(dir).mkdir(parents=True, exist_ok=True)
    json_file = os.path.join(tempfile.gettempdir(), args.output, args.output + '.json')
    options.train_data_dir = os.path.join(tempfile.gettempdir(), args.output)
    res = None

    if args.dreambooth:
        options.in_json = None
    else:
        options.in_json = json_file

    for root, _sub_dirs, folder in os.walk(args.input):
        files = [os.path.join(root, f) for f in folder]

    if not args.noprocess:
        # preprocess
        for f in files:
            try:
                res, metadata = modules.process.process_file(f = f, dst = dir, preview = False, offline = args.offline, txt = args.dreambooth)
                if not args.dreambooth:
                    with open(json_file, "w") as outfile:
                        outfile.write(json.dumps(metadata, indent=2))
            except ValueError as e:
                exit(1)
        modules.process.unload_models()
        mem_stats()
        if args.tag is not None:
            for name, item in metadata.items():
                item['caption'] = args.tag + ',' + item['caption']
                item['tags'] = args.tag + ',' + item['tags']
        with open(json_file, "w") as outfile:
            outfile.write(json.dumps(metadata, indent=2))

    if not args.nolatents and not args.dreambooth:
        # create latents
        latents.create_vae_latents(Map({ 'input': dir, 'json': json_file }))
        latents.unload_vae()
        mem_stats()

        log.info({ 'processed': res, 'inputs': len(files), 'metadata': json_file, 'path': dir })
    else:
        log.info({ 'skip processing': len(files), 'metadata': json_file, 'path': dir })

    if args.shutdown:
        log.info({ 'server shutdown required': True })
        modules.sdapi.shutdown()
        time.sleep(1)

    if not args.notrain:
        train(options)
        mem_stats()
