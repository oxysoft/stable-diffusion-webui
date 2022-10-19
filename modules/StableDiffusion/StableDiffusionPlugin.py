# some of those options should not be changed at all because they would break the model, so I removed them from options.
from collections import namedtuple
from datetime import datetime

from jobs import *
from SDJob import SDJob
from SDJob_img2img import SDJob_img2img
from SDJob_txt2img import SDJob_txt2img
from job_result import JobResult
from lowvram import setup_for_low_vram
from modules.StableDiffusion import CheckpointInfo
from modules.StableDiffusion.CheckpointLoader import CheckpointLoader
from sampling_k import KDiffusionSampler
from sampling_vanilla import VanillaStableDiffusionSampler
from textual_inversion import textual_inversion
from util import *
from ldm.util import instantiate_from_config
from core.modellib import *
from core.installing import git_clone, move_files
from core.options import *
from core.paths import repodir, rootdir, modeldir
from core.plugins import Plugin

from omegaconf import OmegaConf

from core import promptlib, devicelib

import config
from hijack import model_hijack


# def plaintext_to_html(text):
#     text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
#     return text


def get_state_dict_from_checkpoint(pl_sd):
    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]

    return pl_sd




class StableDiffusionPlugin(Plugin):
    def __init__(self, filename=None):
        super().__init__(filename)
        self.model = None
        self.checkpoints = None
        self.hypernetworks_loaded = None
        self.hypernetworks_info = None
        self.samplers = []
        self.samplers_for_img2img = []  # Samplers only for img2img
        self.parallel_processing_allowed = False
        self.all_samplers = []

    def title(self):
        return "The official StableDiffusion plugin, adapted from AUTOMATIC1111's code."

    def options(self, tpl):
        tpl.update(options_section(('sd', "Stable Diffusion"), {
            "sd_hypernetwork"                    : OptionInfo("None", "Stable Diffusion finetune hypernetwork", gradio.Dropdown, lambda: {"choices": ["None"] + [x for x in self.hypernetworks_info.keys()]}),
            "img2img_color_correction"           : OptionInfo(False, "Apply color correction to img2img results to match original colors."),
            "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
            "img2img_fix_steps"                  : OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
            "enable_quantization"                : OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
            "enable_emphasis"                    : OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention"),
            "use_old_emphasis_implementation"    : OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
            "enable_batch_seeds"                 : OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
            "comma_padding_backtrack"            : OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gradio.Slider, {"minimum": 0, "maximum": 74, "step": 1}),
            "filter_nsfw"                        : OptionInfo(False, "Filter NSFW content"),
            'CLIP_stop_at_last_layers'           : OptionInfo(1, "Stop At last layers of CLIP model", gradio.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
        }))

        tpl.update(options_section(('sampler-params', "Sampler parameters"), {
            "hide_samplers"       : OptionInfo([], "Hide samplers in user interface (requires restart)", gradio.CheckboxGroup, lambda: {"choices": [x.name for x in self.all_samplers]}),
            "eta_ddim"            : OptionInfo(0.0, "eta (noise multiplier) for DDIM", gradio.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
            "eta_ancestral"       : OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers", gradio.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
            'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gradio.Number, {"precision": 0}),
        }))

        # shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.StableDiffusionPlugin_hypernetworks.load_hypernetwork(shared.opts.sd_hypernetwork)))

    def install(self):
        stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
        k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "f4e99857772fc3a126ba886aadf795a332774878")

        git_clone("https://github.com/CompVis/stable-diffusion.git", repodir / 'stable-diffusion', "Stable Diffusion", stable_diffusion_commit_hash)
        git_clone("https://github.com/crowsonkb/k-diffusion.git", repodir / 'k-diffusion', "K-diffusion", k_diffusion_commit_hash)

    def args(self, parser):
        # TODO this prob shouldn't be here
        # TODO why do we need to support 3 different ways of specifying the repo??

        sd_path = None
        possible_paths = [Path(rootdir) / 'repositories/stable-diffusion',
                          Path(rootdir),
                          Path(rootdir).parent]

        for path in possible_paths:
            if (path / 'ldm/models/diffusion/ddpm.py').exists():
                sd_path = path
                break

        assert sd_path is not None, f"Couldn't find Stable Diffusion in any of: {possible_paths}"

        parser.add_argument("--config", type=str, default=(sd_path / "configs/stable-diffusion/v1-inference.yaml").as_posix(), help="path to config which constructs model")
        parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded", )
        parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to directory with stable diffusion checkpoints")
        parser.add_argument("--enable-console-prompts", action='store_true', help="print prompts to console when generating with txt2img and img2img", default=False)

    def create_sampler_with_index(self, list_of_configs, index, model):
        config = list_of_configs[index]
        sampler = config.constructor(model)
        sampler.configpath = config

        return sampler

    def load(self):
        import k_diffusion.sampling
        import ldm.models.diffusion.ddim
        import ldm.models.diffusion.plms
        ldm.models.diffusion.ddim.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)
        ldm.models.diffusion.plms.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)

        self.parallel_processing_allowed = not cargs.lowvram and not cargs.medvram

        samplers_k_diffusion = [
            ('Euler a', 'sample_euler_ancestral', ['k_euler_a'], {}),
            ('Euler', 'sample_euler', ['k_euler'], {}),
            ('LMS', 'sample_lms', ['k_lms'], {}),
            ('Heun', 'sample_heun', ['k_heun'], {}),
            ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {}),
            ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {}),
            ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {}),
            ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {}),
            ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
            ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras'}),
            ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras'}),
        ]

        # All sampler data, both k-diffusion + ddpm
        SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])
        self.all_samplers = [
            *[
                SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                for label, funcname, aliases, options in samplers_k_diffusion
                if hasattr(k_diffusion.sampling, funcname)
            ],
            SamplerData('DDIM', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.ddim.DDIMSampler, model), [], {}),
            SamplerData('PLMS', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.plms.PLMSSampler, model), [], {}),
        ]
        self.set_samplers()
        self.set_samplers()

        self.checkpoints = CheckpointLoader(config.MODEL_SUBDIRNAME, cargs.configpath)  # TODO do not use cargs
        self.checkpoints.reload()
        if Path(cargs.ckpt).is_file():
            self.checkpoints.add_file(cargs.ckpt)

        self.load_model(self.checkpoints.get_default())

    def unload(self, args):
        move_files(modeldir, modeldir / config.MODEL_SUBDIRNAME, ".ckpt")  # TODO idk what this is for

    def on_step_start(self, latent):
        pass

    def on_step_condfn(self):
        pass

    def on_step_end(self):
        pass

    def jobs(self):
        return dict(txt2img=self.txt2img,
                    img2img=self.img2img,
                    train_hn=self.train_hn)

    def set_samplers(self):
        hidden = set(opts.hide_samplers)
        hidden_img2img = set(opts.hide_samplers + ['PLMS'])

        self.samplers = [x for x in self.all_samplers if x.name not in hidden]
        self.samplers_for_img2img = [x for x in self.all_samplers if x.name not in hidden_img2img]

    def list_checkpoints(self):
        """
        Returns a list of checkpoint names
        """
        return [x.title for x in self.checkpoints.all]

    def load_model(self, ickpt:CheckpointInfo):
        if ickpt.configpath != cargs.configpath:
            print(f"Loading SD from: {ickpt.configpath}")

        model = instantiate_from_config(OmegaConf.load_py(ickpt.configpath).model)
        self.load_model_weights(model, ickpt)

        # Low VRAM opt
        if cargs.lowvram or cargs.medvram:
            setup_for_low_vram(model, cargs.medvram)
        else:
            model.to(devicelib.device)

        model_hijack.hijack(model)
        model.eval()
        model.info = ickpt

        print(f"Model loaded.")
        return model

    def load_weights(self, model, ickpt=None):
        # CPU opt (wat)
        if cargs.lowvram or cargs.medvram:
            send_everything_to_cpu()
        else:
            model.to(devicelib.cpu)

        model_hijack.undo_hijack(model)
        self.load_model_weights(model, ickpt)
        model_hijack.hijack(model)

        model.info = ickpt
        if not cargs.lowvram and not cargs.medvram:
            model.to(devicelib.device)

        print(f"Weights loaded.")
        return model

    def load_model_weights(self, model, ickpt):
        """
        Load weights from a CheckpointInfo into the instantiated model.
        """
        print(f"Loading weights [{ickpt.hash}] from {ickpt.path}")

        pl = torch.load(ickpt.path, map_location="cpu")
        if "global_step" in pl:
            print(f"Global Step: {pl['global_step']}")

        sd = get_state_dict_from_checkpoint(pl)

        model.load_state_dict(sd, strict=False)

        # Memory Optimizations
        if cargs.opt_channelslast:
            model.to(memory_format=torch.channels_last)
        if not cargs.no_half:
            model.half()

        # why is this here
        devicelib.dtype = torch.float32 if cargs.no_half else torch.float16
        devicelib.dtype_vae = torch.float32 if cargs.no_half or cargs.no_half_vae else torch.float16

        # VAE loading ...................................
        vaepath = ickpt.path.with_suffix(".vae.pt")
        if not os.path.exists(vaepath) and cargs.vae_path is not None:
            vaepath = cargs.vae_path

        if os.path.exists(vaepath):
            print(f"Loading VAE weights from: {vaepath}")
            vae_ckpt = torch.load(vaepath, map_location="cpu")
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}

            model.first_stage_model.load_state_dict(vae_dict)

        model.first_stage_model.to(devicelib.dtype_vae)

    def txt2img(self, p: SDJob_txt2img):
        self.new_job('txt2img', p)
        return self.process_job(p)

    def img2img(self, p: SDJob_img2img):
        # if is_batch:
        #     assert not cargs.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        # fix_seed(p)
        #
        # print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")
        #
        #
        # state.job_count = len(images) * p.n_iter
        #
        # for i, image in enumerate(images):
        #     state.job = f"{i + 1} out of {len(images)}"
        #     if state.skipped:
        #         state.skipped = False
        #
        #     if state.interrupted:
        #         break
        #
        #     img = Image.open(image)
        #     p.init_images = [img] * p.batch_size
        #
        #     result = JobResult(p, [], p.seed, "")  # TODO we should have the whole batch's results in one
        # else:

        p.do_not_save_grid = True

        self.new_job('txt2img', p)
        return self.process_job(p)

    def store_latent(self, decoded, job):
        job.latent = decoded

        if opts.show_progress_every_n_steps > 0 and job.step % opts.show_progress_every_n_steps == 0:
            if not self.parallel_processing_allowed:
                job.image = self.sample_to_image(decoded)

    def decode_first_stage(self, model, x):
        with devicelib.autocast(disable=x.dtype == devicelib.dtype_vae):
            x = model.decode_first_stage(x)

        return x

    def sample_to_image(self, samples):
        x_sample = self.decode_first_stage(self.model, samples[0:1])[0]
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)

        return Image.fromarray(x_sample)

    def process_job(self, p: SDJob) -> JobResult:
        """
        this is the main loop that both txt2img and img2img use;
        it calls func_init once inside all the scopes and func_sample once per batch
        """

        if type(p.prompt) == list:
            assert (len(p.prompt) > 0)
        else:
            assert p.prompt is not None

        devicelib.torch_gc()

        seed = get_fixed_seed(p.seed)
        subseed = get_fixed_seed(p.subseed.seed)

        model_hijack.apply_circular(p.tiling)
        model_hijack.clear_comments()

        comments = {}

        # shared.prompt_styles.apply_styles(p)

        # Prepare the seeds ......................
        if type(p.prompt) == list:
            all_prompts = p.prompt
        else:
            all_prompts = p.batch_size * p.n_iter * [p.prompt]

        if type(seed) == list:
            all_seeds = seed
        else:
            all_seeds = [int(seed) + (x if p.subseed.strength == 0 else 0) for x in range(len(all_prompts))]

        if type(subseed) == list:
            all_subseeds = subseed
        else:
            all_subseeds = [int(subseed) + x for x in range(len(all_prompts))]

        # TODO get embeddings from self.embeddings, or implement them with an hijack plugin? idk
        # if Path(cargs.embeddings_dir).exists():
        #     model_hijack.embedding_db.load_textual_inversion_embeddings()

        # infotexts = []
        # def infotext(iteration=0, position_in_batch=0):
        #     return create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration, position_in_batch)

        # Start rendering ...........................................
        from core.devicelib import autocast

        output_images = []

        with torch.no_grad(), p.plugin.model.ema_scope():
            # Init the job
            with autocast():
                p.init(all_prompts, all_seeds, all_subseeds)

            # Execute the job
            for n in range(p.n_iter):  # I dont recommend using this but ok
                if p.job.aborted:
                    break

                # Prepare the seeds .............................
                prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
                subseeds = all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

                if len(prompts) == 0:
                    break

                # Prompt conditionings.................................
                # uc = p.sd_model.get_learned_conditioning(len(prompts) * [p.negative_prompt])
                # c = p.sd_model.get_learned_conditioning(prompts)
                with autocast():
                    uc = promptlib.get_learned_conditioning(self.model, len(prompts) * [p.promptneg], p.steps)
                    c = promptlib.get_multicond_learned_conditioning(self.model, prompts, p.steps)

                # Wtf is this??????
                if len(model_hijack.comments) > 0:
                    for comment in model_hijack.comments:
                        comments[comment] = 1

                # Sampling -----------------------------------
                with autocast():
                    samples_ddim = p.sample(conditioning=c,
                                            unconditional_conditioning=uc,
                                            seeds=seeds,
                                            subseeds=subseeds,
                                            subseed_strength=p.subseed.strength)

                # Use what's done
                if p.job.aborted:
                    samples_ddim = p.job.latent

                samples_ddim = samples_ddim.to(devicelib.dtype_vae)
                x_samples_ddim = self.decode_first_stage(p.plugin.model, samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                del samples_ddim

                if cargs.lowvram or cargs.medvram:
                    send_everything_to_cpu()

                devicelib.torch_gc()
                del x_samples_ddim
                devicelib.torch_gc()

            # index_of_first_image = 0
            # unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
            # if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            #     grid = imagelib.image_grid(output_images, p.batch_size)
            #
            #     if opts.return_grid:
            #         text = infotext()
            #         infotexts.insert(0, text)
            #         if opts.enable_pnginfo:
            #             grid.info["parameters"] = text
            #         output_images.insert(0, grid)
            #         index_of_first_image = 1
            #
            #     if opts.grid_save:
            #         imagelib.save_image(grid, p.outpath_grids, "grid", all_seeds[0], all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

        devicelib.torch_gc()
        return JobResult(p,
                         output_images,
                         all_seeds[0],
                         infotext() + "".join(["\n\n" + x for x in comments]),
                         subseed=all_subseeds[0],
                         all_prompts=all_prompts,
                         all_seeds=all_seeds,
                         all_subseeds=all_subseeds,
                         index_of_first_image=index_of_first_image,
                         infotexts=infotexts)

    def get_correct_sampler(self, p):
        if isinstance(p, SDJob_txt2img):
            return self.samplers
        elif isinstance(p, SDJob_img2img):
            return self.samplers_for_img2img

    def train_hn(self,
                 p: SDJob,
                 hypernetwork_name,
                 learn_rate,
                 data_root,
                 log_directory, steps, create_image_every, save_hypernetwork_every,
                 template_file,
                 preview_image_prompt):
        job = self.new_job("train_hn", p)
        job.state = "Initializing hypernetwork training..."
        job.stepmax = steps

        path = self.hypernetworks_info.get(hypernetwork_name, None)

        hypernetworks_loaded = Hypernetwork()
        hypernetworks_loaded.load_py(path)

        filename = os.path.join(cargs.hypernetwork_dir, f'{hypernetwork_name}.pt')

        log_directory = os.path.join(log_directory, datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
        unload = opts.unload_models_when_training

        if save_hypernetwork_every > 0:
            hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
            os.makedirs(hypernetwork_dir, exist_ok=True)
        else:
            hypernetwork_dir = None

        if create_image_every > 0:
            images_dir = os.path.join(log_directory, "images")
            os.makedirs(images_dir, exist_ok=True)
        else:
            images_dir = None

        # shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
        job.state = f"Preparing dataset from {data_root}..."
        with torch.autocast("cuda"):
            ds = textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=512, height=512, repeats=1, placeholder_token=hypernetwork_name, model=sd.model, device=devicelib.device, template_file=template_file, include_cond=True)

        if unload:
            sd.model.cond_stage_model.to(devicelib.cpu)
            sd.model.first_stage_model.to(devicelib.cpu)

        hypernetwork = hypernetworks_loaded
        weights = hypernetwork.weights()
        for weight in weights:
            weight.requires_grad = True

        losses = torch.zeros((32,))

        last_saved_file = "<none>"
        last_saved_image = "<none>"

        ititial_step = hypernetwork.step or 0
        if ititial_step > steps:
            return hypernetwork, filename

        schedules = iter(LearnSchedule(learn_rate, steps, ititial_step))
        (learn_rate, end_step) = next(schedules)
        print(f'Training at rate of {learn_rate} until step {end_step}')

        optimizer = torch.optim.AdamW(weights, lr=learn_rate)

        pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
        for i, (x, text, cond) in pbar:
            hypernetwork.step = i + ititial_step

            if hypernetwork.step > end_step:
                try:
                    (learn_rate, end_step) = next(schedules)
                except Exception:
                    break
                tqdm.tqdm.write(f'Training at rate of {learn_rate} until step {end_step}')
                for pg in optimizer.param_groups:
                    pg['lr'] = learn_rate

            if job.interrupted:
                break

            with torch.autocast("cuda"):
                cond = cond.to(devicelib.device)
                x = x.to(devicelib.device)
                loss = sd.model(x.unsqueeze(0), cond)[0]
                del x
                del cond

                losses[hypernetwork.step % losses.shape[0]] = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.set_description(f"loss: {losses.mean():.7f}")

            if hypernetwork.step > 0 and hypernetwork_dir is not None and hypernetwork.step % save_hypernetwork_every == 0:
                last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork_name}-{hypernetwork.step}.pt')
                hypernetwork.save(last_saved_file)

            if hypernetwork.step > 0 and images_dir is not None and hypernetwork.step % create_image_every == 0:
                last_saved_image = os.path.join(images_dir, f'{hypernetwork_name}-{hypernetwork.step}.png')

                preview_text = text if preview_image_prompt == "" else preview_image_prompt

                # TODO call StableDiffusion t2i
                # optimizer.zero_grad()
                # sd.model.cond_stage_model.to(devicelib.device)
                # sd.model.first_stage_model.to(devicelib.device)

                # p = StableDiffusionPlugin_processing_t2i.StableDiffusionProcessingTxt2Img(
                #         sd_model=sd.model,
                #         prompt=preview_text,
                #         steps=20,
                #         do_not_save_grid=True,
                #         do_not_save_samples=True,
                # )
                #
                # processed = processing.process_images(p)
                # image = processed.images[0]

                # if unload:
                #     sd.model.cond_stage_model.to(devicelib.cpu)
                #     sd.model.first_stage_model.to(devicelib.cpu)
                #
                # shared.state.current_image = image
                # image.save(last_saved_image)
                #
                # last_saved_image += f", prompt: {preview_text}"

        #         shared.state.textinfo = f"""
        # <p>
        # Loss: {losses.mean():.7f}<br/>
        # Step: {hypernetwork.step}<br/>
        # Last prompt: {html.escape(text)}<br/>
        # Last saved embedding: {html.escape(last_saved_file)}<br/>
        # Last saved image: {html.escape(last_saved_image)}<br/>
        # </p>
        # """
        #
        checkpoint = self.checkpoints.get_default()

        hypernetwork.sd_checkpoint = checkpoint.hash
        hypernetwork.sd_checkpoint_name = checkpoint.model_name
        hypernetwork.save(filename)

        return hypernetwork, filename


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetwork(hypernetworks_loaded, context, self)
    k = self.to_k(context_k)
    v = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


# TODO Handle in CensorPlugin
# if opts.filter_nsfw:
#     x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)

# TODO post img to session manager
# for i, x_sample in enumerate(x_samples_ddim):
#     x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
#     x_sample = x_sample.astype(np.uint8)
#
#     # TODO handle in FaceRestorePlugin
#     if p.restore_faces:
#         if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
#             imagelib.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")
#
#         devices.torch_gc()
#
#         x_sample = modules.face_restoration.restore_faces(x_sample)
#         devices.torch_gc()
#
#     image = Image.fromarray(x_sample)
#
#     # TODO handle in ColorCorrectionPlugin
#     if p.color_corrections is not None and i < len(p.color_corrections):
#         if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
#             imagelib.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
#         image = apply_color_correction(p.color_corrections[i], image)
#
#     # TODO handle overlay images plugin (idk what this is)
#     if p.overlay_images is not None and i < len(p.overlay_images):
#         overlay = p.overlay_images[i]
#
#         if p.paste_to is not None:
#             x, y, w, h = p.paste_to
#             base_image = Image.new('RGBA', (overlay.width, overlay.height))
#             image = imagelib.resize_image(1, image, w, h)
#             base_image.paste(image, (x, y))
#             image = base_image
#
#         image = image.convert('RGBA')
#         image.alpha_composite(overlay)
#         image = image.convert('RGB')
#
#     if opts.samples_save and not p.do_not_save_samples:
#         imagelib.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)
#
#     text = infotext(n, i)
#     infotexts.append(text)
#     if opts.enable_pnginfo:
#         image.info["parameters"] = text
#     output_images.append(image)
def extended_tdqm(sequence, *args, desc=None, **kwargs):
    job.sampling_steps = len(sequence)
    job.sampling_step = 0

    seq = sequence \
        if cargs.disable_console_progressbars \
        else tqdm.tqdm(sequence, *args, desc=job.job, file=progress_print_out, **kwargs)

    for x in seq:
        if job.aborted or job.skipped:
            break

        yield x

        job.update_step()
