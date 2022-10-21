# some of those options should not be changed at all because they would break the model, so I removed them from options.
from collections import namedtuple
from datetime import datetime
from enum import Enum

import tqdm

from core.jobs import Job
from SDJob import SDJob
from SDJob_img2img import SDJob_img2img
from SDJob_txt2img import SDJob_txt2img
from SDJob_train_embedding import SDJob_train_embedding
from SDCheckpointLoader import SDCheckpointLoader
from optimizations import invokeAI_mps_available
from Hypernetwork import Hypernetwork
from TextInvDataset import TextInvDataset
from TextinvLearnSchedule import TextinvLearnSchedule
from SDSampler_K import SDSampler_K
from SDSampler_Vanilla import SDSampler_Vanilla
from util import *
from core.modellib import *
from core.installing import git_clone, move_files
from core.options import *
from core.paths import repodir, modeldir
from core.plugins import Plugin

from core import promptlib, devicelib, paths

import SDConstants

# TODO we can't do this here, we must instantiate the plugin without our installations done
import ldm
import ldm.modules.attention
import ldm.modules.diffusionmodules.model

ldm_crossattention_forward = ldm.modules.attention.CrossAttention.forward
ldm_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
ldm_attnblock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward


# def plaintext_to_html(text):
#     text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
#     return text

class StableDiffusionAttention(Enum):
    LDM = 0
    SPLIT_V1 = 1
    SPLIT_INVOKE = 2
    SPLIT_DOGGETT = 3
    XFORMERS = 4


class StableDiffusionPlugin(Plugin):
    class Options:
        def __init__(self, plugin):
            # tpl.update(options_section(('sd', "Stable Diffusion"), {
            #     "sd_hypernetwork"                    : OptionInfo("None", "Stable Diffusion finetune hypernetwork", gradio.Dropdown, lambda: {"choices": ["None"] + [x for x in self.hypernetworks_info.keys()]}),
            #     "img2img_color_correction"           : OptionInfo(False, "Apply color correction to img2img results to match original colors."),
            #     "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
            #     "img2img_fix_steps"                  : OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
            #     "enable_quantization"                : OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
            #     "enable_emphasis"                    : OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention"),
            #     "use_old_emphasis_implementation"    : OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
            #     "enable_batch_seeds"                 : OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
            #     "comma_padding_backtrack"            : OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gradio.Slider, {"minimum": 0, "maximum": 74, "step": 1}),
            #     "filter_nsfw"                        : OptionInfo(False, "Filter NSFW content"),
            #     'CLIP_stop_at_last_layers'           : OptionInfo(1, "Stop At last layers of CLIP model", gradio.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
            # }))
            #
            # tpl.update(options_section(('sampler-params', "Sampler parameters"), {
            #     "hide_samplers"       : OptionInfo([], "Hide samplers in user interface (requires restart)", gradio.CheckboxGroup, lambda: {"choices": [x.name for x in self.all_samplers]}),
            #     "eta_ancestral"       : OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers", gradio.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
            #     'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gradio.Number, {"precision": 0}),
            # }))

            # Defaults
            self.configpath = plugin.get_repo_dir() / "configs/stable_diffusion/v1-inference.yaml"
            self.ckptpath = ""
            self.lowvram = False
            self.medvram = False
            self.opt_channelslast = False
            self.no_half = False
            self.no_half_vae = False
            self.vae_override = ""

            self.attention = StableDiffusionAttention.LDM
            # self.opt_split_attention_invokeai = None
            # self.opt_split_attention_v1 = None
            # self.disable_opt_split_attention = None

    def __init__(self, dirpath=None):
        super().__init__(dirpath)
        self.opt = StableDiffusionPlugin.Options(self)  # TODO some way to store and load this
        self.checkpoints = SDCheckpointLoader(
                self,
                constants.model_dirname,
                self.opt.configpath,
                ["model", "sd-v1-4"])
        self.model = None  # Model currently in use, loaded through SDCheckpointLoader
        self.samplers = []  # List of SamplerData we can use
        # self.embeddings = StableDiffusionEmbeddingCollection(
        #         self,
        #         config.model_dirname,
        #         self.opt.configpath,
        #         ["model", "sd-v1-4"])

    def title(self):
        return "The official stable_diffusion plugin, adapted from AUTOMATIC1111's code."

    def install(self):
        stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
        k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "f4e99857772fc3a126ba886aadf795a332774878")

        git_clone("https://github.com/CompVis/stable-diffusion.git", repodir / 'stable_diffusion', "Stable Diffusion", stable_diffusion_commit_hash)
        git_clone("https://github.com/crowsonkb/k-diffusion.git", repodir / 'k-diffusion', "K-diffusion", k_diffusion_commit_hash)

        # Idk what is the point of this code, it should be installed in one place......... we only need one valid place TODO
        repo = self.get_repo_dir()
        assert repo.is_dir() is not None, f"Couldn't find Stable Diffusion in {repo}"

        # TODO install xformers if enabled in opts
        if self.opt.attention == StableDiffusionAttention.XFORMERS:
            import xformers.ops

        # TODO install mps if invoke attention

    def get_repo_dir(self) -> Path:
        return repodir / "stable_diffusion"

    def allow_parallel_processing(self):
        return not self.opt.lowvram and not self.opt.medvram

    # noinspection PyUnresolvedReferences
    def load(self):
        import k_diffusion.sampling
        import ldm.models.diffusion.ddim
        import ldm.models.diffusion.plms
        # ldm.models.diffusion.ddim.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)
        # ldm.models.diffusion.plms.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)

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
        self.samplers = [
            *[
                SamplerData(label, lambda model, funcname=funcname: SDSampler_K(funcname, model), aliases, options)
                for label, funcname, aliases, options in samplers_k_diffusion
                if hasattr(k_diffusion.sampling, funcname)
            ],
            SamplerData('DDIM', lambda model: SDSampler_Vanilla(ldm.models.diffusion.ddim.DDIMSampler, model), [], {}),
            SamplerData('PLMS', lambda model: SDSampler_Vanilla(ldm.models.diffusion.plms.PLMSSampler, model), [], {}),
        ]

        self.checkpoints.reload()
        self.model = self.checkpoints.load(self.checkpoints.get_default())

    def unload(self):
        move_files(paths.modeldir, paths.modeldir / SDConstants.model_dirname, ".ckpt")  # TODO idk what this is for

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

    def list_checkpoints(self):
        """
        Returns a list of checkpoint names
        """
        return [x.title for x in self.checkpoints.all]

    def txt2img(self, p: SDJob_txt2img):
        j = self.new_job('txt2img', p)
        self.gen(j)

    def img2img(self, p: SDJob_img2img):
        fix_seed(p)
        j = self.new_job('txt2img', p)
        self.gen(j)

        # print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    def reset_ldm_overrides(self):
        ldm.modules.attention.CrossAttention.forward = ldm_crossattention_forward
        ldm.modules.diffusionmodules.model.nonlinearity = ldm_nonlinearity
        ldm.modules.diffusionmodules.model.AttnBlock.forward = ldm_attnblock_forward

    def set_ldm_overrides(self):
        # noinspection PyUnresolvedReferences
        from optimizations import \
            xformers_attention_forward, \
            xformers_attnblock_forward, \
            split_cross_attention_forward_v1, \
            split_cross_attention_forward_invokeAI, \
            split_cross_attention_forward, \
            cross_attention_attnblock_forward

        ldm.modules.diffusionmodules.model.nonlinearity = torch.nn.functional.silu
        mode = self.opt.attention

        if not invokeAI_mps_available and devicelib.device.type == 'mps':
            print("Cannot use InvokeAI cross attention optimization for MPS without psutil package, which is not installed.")
            print("Reverting to LDM.")
            mode = StableDiffusionAttention.LDM

        if mode == StableDiffusionAttention.XFORMERS:
            if not (torch.version.cuda and (6, 0) <= torch.cuda.get_device_capability(devicelib.device) <= (8, 6)):
                print("Cannot use xformers attention with the current CUDA version or GPU. Reverting to LDM")
                mode = StableDiffusionAttention.LDM

        if mode == StableDiffusionAttention.XFORMERS and torch.version.cuda and (6, 0) <= torch.cuda.get_device_capability(devicelib.device) <= (8, 6):
            print("Applying xformers cross attention optimization.")
            ldm.modules.attention.CrossAttention.forward = xformers_attention_forward
            ldm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward
        elif mode == StableDiffusionAttention.SPLIT_V1:
            print("Applying v1 cross attention optimization.")
            ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1
        elif mode == StableDiffusionAttention.SPLIT_INVOKE or not torch.cuda.is_available():
            print("Applying cross attention optimization (InvokeAI).")
            ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI
        elif mode == StableDiffusionAttention.SPLIT_DOGGETT:
            print("Applying cross attention optimization (Doggettx).")
            ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
            ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward

    def gen(self, job: Job):
        """
        this is the main loop that both txt2img and img2img use;
        it calls func_init once inside all the scopes and func_sample once per batch
        """
        p: SDJob = job.p

        if type(p.prompt) == list:
            assert (len(p.prompt) > 0)
        else:
            assert p.prompt is not None

        devicelib.torch_gc()

        seed = get_fixed_seed(p.seed)
        subseed = get_fixed_seed(p.subseed.seed)

        self.checkpoints.set_circular(self.model, p.tiling)

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
                if job.aborted:
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
                if len(self.model.comments) > 0:
                    for comment in self.model.comments:
                        comments[comment] = 1

                # Sampling -----------------------------------
                with autocast():
                    samples_ddim = p.sample(conditioning=c,
                                            unconditional_conditioning=uc,
                                            seeds=seeds,
                                            subseeds=subseeds,
                                            subseed_strength=p.subseed.strength)

                # Use what's done
                if job.aborted:
                    samples_ddim = job.latent

                samples_ddim = samples_ddim.to(devicelib.dtype_vae)
                x_samples_ddim = p.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                del samples_ddim

                if self.opt.lowvram or self.opt.medvram:
                    send_everything_to_cpu()

                devicelib.torch_gc()
                del x_samples_ddim
                devicelib.torch_gc()

        devicelib.torch_gc()
        # return JobResult(p,
        #                  output_images,
        #                  all_seeds[0],
        #                  infotext() + "".join(["\n\n" + x for x in comments]),
        #                  subseed=all_subseeds[0],
        #                  all_prompts=all_prompts,
        #                  all_seeds=all_seeds,
        #                  all_subseeds=all_subseeds,
        #                  index_of_first_image=index_of_first_image,
        #                  infotexts=infotexts)

    # def get_correct_sampler(self, p):
    #     if isinstance(p, SDJob_txt2img):
    #         return self.samplers
    #     elif isinstance(p, SDJob_img2img):
    #         return self.samplers_for_img2img

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
        hypernetworks_loaded.load(path)

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
        with torch.autocast("cuda"):
            ds = TextInvDataset(data_root=data_root,
                                width=512,
                                height=512,
                                repeats=1,
                                placeholder_token=hypernetwork_name,
                                model=p.model,
                                device=devicelib.device,
                                template_file=template_file,
                                include_cond=True)

        if unload:
            p.model.cond_stage_model.to(devicelib.cpu)
            p.model.first_stage_model.to(devicelib.cpu)

        hypernetwork = hypernetworks_loaded
        weights = hypernetwork.weights()
        for weight in weights:
            weight.requires_grad = True

        losses = torch.zeros((32,))

        last_saved_file = "<none>"
        last_saved_image = "<none>"

        initsteps = hypernetwork.step or 0
        if initsteps > steps:
            return hypernetwork, filename

        schedules = iter(TextinvLearnSchedule(learn_rate, steps, initsteps))
        (learn_rate, end_step) = next(schedules)

        print(f'Training at rate of {learn_rate} until step {end_step}')

        optimizer = torch.optim.AdamW(weights, lr=learn_rate)
        pbar = tqdm.tqdm(enumerate(ds), total=steps - initsteps)
        for i, (x, text, cond) in pbar:
            hypernetwork.step = i + initsteps

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
                loss = self.model(x.unsqueeze(0), cond)[0]
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

                # TODO call stable_diffusion t2i
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

    def train_embedding(self, p: SDJob_train_embedding):
        assert p.name, 'embedding not selected'

        # TODO this is a job

        j = self.new_job("train_embedding", p)
        j.update("Initializing textual inversion training...")
        j.stepmax = p.steps

        path = paths.embeddingdir.embeddings_dir / f'{p.name}.pt'  # TODO need a proper path

        # log_directory = os.path.join(p.log_directory, datetime.now().strftime("%Y-%m-%d"), p.embedding_name)

        cond_model = self.model.cond_stage_model

        j.state = f"Preparing dataset from {p.datadir}..."
        with torch.autocast("cuda"):
            ds = TextInvDataset(data_root=p.datadir,
                                width=p.training_width,
                                height=p.training_height,
                                repeats=p.num_repeats,
                                placeholder_token=p.name,
                                model=p.model,
                                device=devicelib.device,
                                template_file=p.template_file)

        embedding = hijack.embedding_db.word_embeddings[p.name]
        embedding.vec.requires_grad = True

        losses = torch.zeros((32,))

        last_saved_file = "<none>"
        last_saved_image = "<none>"

        ititial_step = embedding.step or 0
        if ititial_step > p.steps:
            return embedding, path

        schedules = iter(TextinvLearnSchedule(p.learn_rate, p.steps, ititial_step))
        (learn_rate, end_step) = next(schedules)
        print(f'Training at rate of {learn_rate} until step {end_step}')

        optimizer = torch.optim.AdamW([embedding.vec], lr=learn_rate)

        pbar = tqdm.tqdm(enumerate(ds), total=p.steps - ititial_step)
        for i, (x, text, _) in pbar:
            embedding.step = i + ititial_step

            if embedding.step > end_step:
                try:
                    (learn_rate, end_step) = next(schedules)
                except:
                    break
                tqdm.tqdm.write(f'Training at rate of {learn_rate} until step {end_step}')
                for pg in optimizer.param_groups:
                    pg['lr'] = learn_rate

            if j.aborted:
                break

            with torch.autocast("cuda"):
                c = cond_model([text])

                x = x.to(devicelib.device)
                loss = p.shared.sd_model(x.unsqueeze(0), c)[0]
                del x

                losses[embedding.step % losses.shape[0]] = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_num = embedding.step // len(ds)
            epoch_step = embedding.step - (epoch_num * len(ds)) + 1

            pbar.set_description(f"[Epoch {epoch_num}: {epoch_step}/{len(ds)}]loss: {losses.mean():.7f}")

            if embedding.step > 0 and embedding_dir is not None and embedding.step % p.save_embedding_every == 0:
                last_saved_file = os.path.join(embedding_dir, f'{p.name}-{embedding.step}.pt')
                embedding.save(last_saved_file)

            if embedding.step > 0 and images_dir is not None and embedding.step % p.create_image_every == 0:
                last_saved_image = os.path.join(images_dir, f'{p.name}-{embedding.step}.png')

                preview_text = text if p.preview_image_prompt == "" else p.preview_image_prompt

                p = p.SDJob_txt2img.SDJob_txt2img(
                        sd_model=p.shared.sd_model,
                        prompt=preview_text,
                        steps=20,
                        height=p.training_height,
                        width=p.training_width,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                )

                processed = p.processing.process_job(p)
                image = processed.images[0]

                p.shared.state.current_image = image

                if p.save_image_with_stored_embedding and os.path.exists(last_saved_file):
                    last_saved_image_chunks = os.path.join(images_embeds_dir, f'{p.name}-{embedding.step}.png')

                    info = p.PngImagePlugin.PngInfo()
                    data = torch.load(last_saved_file)
                    info.add_text("sd-ti-embedding", p.embedding_to_b64(data))

                    title = "<{}>".format(data.get('name', '???'))
                    checkpoint = p.sd_models.get_checkpoint()
                    footer_left = checkpoint.model_name
                    footer_mid = '[{}]'.format(checkpoint.hash)
                    footer_right = '{}'.format(embedding.step)

                    captioned_image = p.caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                    captioned_image = p.insert_image_data_embed(captioned_image, data)

                    captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)

                image.save(last_saved_image)

                last_saved_image += f", prompt: {preview_text}"

            p.shared.state.job_no = embedding.step

            p.shared.state.textinfo = f"""
    <p>
    Loss: {losses.mean():.7f}<br/>
    Step: {embedding.step}<br/>
    Last prompt: {p.html.escape(text)}<br/>
    Last saved embedding: {p.html.escape(last_saved_file)}<br/>
    Last saved image: {p.html.escape(last_saved_image)}<br/>
    </p>
    """

        checkpoint = p.sd_models.get_checkpoint()

        embedding.sd_checkpoint = checkpoint.hash
        embedding.sd_checkpoint_name = checkpoint.model_name
        embedding.cached_checksum = None
        embedding.save(path)

        return embedding, path

    def on_run_start(self):
        pass

    def on_postprocess_run_params(self, p):
        pass

    def on_batch_start(self, p):
        pass

    def on_postprocess_batch_params(self, p):
        pass

    def on_postprocess_image(self, img):
        """
        Called for every image output of each batch.
        Args:
            img: the output image to process, in cv2 rgb format. Use utilities to convert

        Returns: the modified image.
        """
        pass

    def on_postprocess_path(self, img, path: Path):
        """
        Post-process the save path for an output image.
        Args:
            img: The image in cv2 RGB format.
            path: The path that the image was saved to.

        Returns:
        """

    def on_image_saved(self, img, path: Path):
        """
        An ouput image has been saved.
        Args:
            img: The image in cv2 RGB format.
            path: The path that the image was saved to.

        Returns:
        """

    def on_batch_end(self):
        pass

    def on_run_end(self):
        pass

    def on_run_interrupted(self):
        pass

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

# def extended_tdqm(sequence, *args, desc=None, **kwargs):
#     job.sampling_steps = len(sequence)
#     job.sampling_step = 0
#
#     # seq = sequence \
#     #     if cargs.disable_console_progressbars \
#     #     else tqdm.tqdm(sequence, *args, desc=job.job, file=progress_print_out, **kwargs)
#
#     for x in seq:
#         if job.aborted or job.skipped:
#             break
#
#         yield x
#
#         job.update_step()


# Grid code at the end of process_job
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
