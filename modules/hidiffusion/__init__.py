from modules import shared
from modules.hidiffusion import hidiffusion
# from modules.hidiffusion import hidiffusion_orig as hidiffusion


def apply_hidiffusion(p):
    if hasattr(shared.sd_model, "named_modules"): # Some pipelines doesn't have named_modules
        hidiffusion.remove_hidiffusion(shared.sd_model)
    if p.hidiffusion:
        hidiffusion.is_aggressive_raunet = shared.opts.hidiffusion_steps > 0
        hidiffusion.aggressive_step = shared.opts.hidiffusion_steps
        if shared.opts.hidiffusion_t1 >= 0:
            t1 = shared.opts.hidiffusion_t1
            hidiffusion.switching_threshold_ratio_dict['sd15_1024']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sd15_2048']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sdxl_2048']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sdxl_4096']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sdxl_turbo_1024']['T1_ratio'] = t1
            p.extra_generation_params['HiDiffusion Ratios'] = f'{shared.opts.hidiffusion_t1}/{shared.opts.hidiffusion_t2}'
        if shared.opts.hidiffusion_t2 >= 0:
            t2 =shared.opts.hidiffusion_t2
            hidiffusion.switching_threshold_ratio_dict['sd15_1024']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sd15_2048']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sdxl_2048']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sdxl_4096']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sdxl_turbo_1024']['T2_ratio'] = t2
            p.extra_generation_params['HiDiffusion Ratios'] = f'{shared.opts.hidiffusion_t1}/{shared.opts.hidiffusion_t2}'
        shared.log.debug(f'Applying HiDiffusion: raunet={shared.opts.hidiffusion_raunet} attn={shared.opts.hidiffusion_attn} aggressive={shared.opts.hidiffusion_steps > 0}:{shared.opts.hidiffusion_steps} t1={shared.opts.hidiffusion_t1} t2={shared.opts.hidiffusion_t2}')
        p.extra_generation_params['HiDiffusion'] = f'{shared.opts.hidiffusion_raunet}/{shared.opts.hidiffusion_attn}/{shared.opts.hidiffusion_steps > 0}:{shared.opts.hidiffusion_steps}'
        hidiffusion.apply_hidiffusion(shared.sd_model, apply_raunet=shared.opts.hidiffusion_raunet, apply_window_attn=shared.opts.hidiffusion_attn)


remove_hidiffusion = hidiffusion.remove_hidiffusion
