import numpy as np
import torch
from PIL import ImageOps, ImageFilter
from PIL.Image import Image

from core import imagelib, devicelib
from modules.StableDiffusion.config import opt_C, opt_f
from modules.StableDiffusion.SDJob import SDJob
from modules.StableDiffusion.util import get_crop_region, expand_crop_region, fill, create_random_tensors


class SDJob_img2img(SDJob):
    def __init__(self,
                 init_images=None,
                 resize_mode=0,
                 denoising_strength=0.75,
                 mask=None,
                 mask_blur=4,
                 inpainting_fill=0,
                 inpaint_full_res=True,
                 inpaint_full_res_padding=0,
                 inpainting_mask_invert=0,
                 **kwargs):
        super().__init__(**kwargs)

        # p.extra_generation_params["Mask blur"] = mask_blur # What the fuck is thiat
        # is_inpaint = mode == 1  # TODO OUCH
        # is_batch = mode == 2  # TODO OUCH

        # if is_inpaint:
        #     if mask_mode == 0:
        #         image = init_img_with_mask['image']
        #         mask = init_img_with_mask['mask']
        #         alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        #         mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
        #         image = image.convert('RGB')
        #     else:
        #         image = init_img_inpaint
        #         mask = init_mask_inpaint
        # else:
        #     image = init_img
        #     mask = None

        assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None
        self.image_mask = mask
        # self.image_unblurred_mask = None
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.mask = None
        self.nmask = None

    def init(self, all_prompts, all_seeds, all_subseeds):
        crop_region = None

        if self.image_mask is not None:
            self.image_mask = self.image_mask.convert('L')

            if self.inpainting_mask_invert:
                self.image_mask = ImageOps.invert(self.image_mask)

            # self.image_unblurred_mask = self.image_mask

            if self.mask_blur > 0:
                self.image_mask = self.image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = imagelib.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2 - x1, y2 - y1)
            else:
                self.image_mask = imagelib.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else self.image_mask

        # TODO CC Plugin
        # add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        # if add_color_corrections:
        #     self.color_corrections = []

        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = imagelib.resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = imagelib.resize_image(2, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = fill(image, latent_mask)

            # TODO CC Plugin, most likely doesn't even need to be done here
            # if add_color_corrections:
            #     self.color_corrections.append(setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(devicelib.device)

        self.init_latent = self.plugin.model.get_first_stage_encoding(self.plugin.model.encode_first_stage(image))

        if self.image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(devicelib.device).type(self.plugin.model.dtype)
            self.nmask = torch.asarray(latmask).to(devicelib.device).type(self.plugin.model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed.strength, seed_resize_from_h=self.subseed.resize_from_h, seed_resize_from_w=self.subseed.resize_from_w, p=self)

        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        del x
        devicelib.torch_gc()

        return samples