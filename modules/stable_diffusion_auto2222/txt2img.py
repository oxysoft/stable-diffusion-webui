import scripts
from processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from shared import opts, cmd_opts
import shared as shared
import processing as processing


def txt2img(prompt: str,
            negative_prompt: str = "",
            prompt_style: str = "",
            prompt_style2: str = "",
            steps: int = 22,
            sampler_index: int = 0,
            restore_faces: bool = False,
            tiling: bool = False,
            n_iter: int = 1,
            batch_size: int = 1,
            cfg_scale: float = 7.0,
            seed: int = -1,
            subseed: int = -1,
            subseed_strength: float = 1,
            seed_resize_from_h: int = 512,
            seed_resize_from_w: int = 512,
            seed_enable_extras: bool = False,
            height: int = 512,
            width: int = 512,
            enable_hr: bool = False,
            denoising_strength: float = 0.5,
            firstphase_width: int = 0,
            firstphase_height: int = 0,
            *args):
    p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            # outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            # outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=prompt,
            # styles=[prompt_style, prompt_style2],
            negative_prompt=negative_prompt,
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            seed_enable_extras=seed_enable_extras,
            sampler_index=sampler_index,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg=cfg_scale,
            width=width,
            height=height,
            # restore_faces=restore_faces,
            tiling=tiling,
            enable_hr=enable_hr,
            denoising_strength=denoising_strength if enable_hr else None,
            firstphase_width=firstphase_width if enable_hr else None,
            firstphase_height=firstphase_height if enable_hr else None,
    )

    p.script_args = args

    # if cmd_opts.enable_console_prompts:
    #     print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    processed = process_images(p)

    return processed.images
