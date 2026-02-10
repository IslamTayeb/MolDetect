import os
import argparse
from typing import List
from concurrent.futures import ThreadPoolExecutor
import PIL
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .pix2seq import build_pix2seq_model
from .tokenizer import get_tokenizer
from .dataset import make_transforms
from .data import postprocess_reactions, postprocess_reactions_deferred, postprocess_bboxes, postprocess_coref_results, ReactionImageData, ImageData, CorefImageData, get_rxnscribe_timing, reset_rxnscribe_timing, record_timing

from molscribe import MolScribe
from huggingface_hub import hf_hub_download
import easyocr


class RxnScribe:

    def __init__(self, model_path, device=None, molscribe=None):
        """
        RxnScribe Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        :param molscribe: optional external MolScribe instance (for caching/sharing)
        """
        args = self._get_args()
        args.format = 'reaction'
        states = torch.load(model_path, map_location=torch.device('cpu'))
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.model = self.get_model(args, self.tokenizer, self.device, states['state_dict'])
        self.transform = make_transforms('test', augment=False, debug=False)
        self.molscribe = molscribe if molscribe is not None else self.get_molscribe()
        self.ocr_model = self.get_ocr_model()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')
        # Data
        parser.add_argument('--format', type=str, default='reaction')
        parser.add_argument('--input_size', type=int, default=1333)

        args = parser.parse_args([])
        args.pix2seq = True
        args.pix2seq_ckpt = None
        args.pred_eos = True
        args.is_coco = False
        args.use_hf_transformer = False
        return args

    def get_model(self, args, tokenizer, device, model_states):
        def remove_prefix(state_dict):
            return {k.replace('model.', ''): v for k, v in state_dict.items()}

        model = build_pix2seq_model(args, tokenizer[args.format])
        model.load_state_dict(remove_prefix(model_states), strict=False)
        model.to(device)
        model.eval()
        return model

    def get_molscribe(self):
        ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        molscribe = MolScribe(ckpt_path, device=self.device)
        return molscribe

    def get_ocr_model(self):
        reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'))
        return reader

    def predict_images(self, input_images: List, batch_size=16, molscribe=False, ocr=False, skip_molblock=False):
        # images: a list of PIL images
        import time
        from .data import _get_rxnscribe_timing

        # Reset timing and track overall
        reset_rxnscribe_timing()
        overall_start = time.time()
        timing_data = {
            'modules': [],
            'transform_time': 0,
            'model_inference_time': 0,
            'postprocess_time': 0,
            'molscribe_time': 0,
            'ocr_time': 0,
            'ocr_call_count': 0,
        }

        device = self.device
        tokenizer = self.tokenizer['reaction']

        # Phase 1: Batch model inference + sequence parsing for ALL images
        all_reaction_sets = []  # List of (img_idx, ReactionSet)

        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]

            # Transform timing
            t0 = time.time()
            images, refs = zip(*[self.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(device)
            timing_data['transform_time'] += time.time() - t0

            # Model inference timing
            t0 = time.time()
            with torch.no_grad():
                pred_seqs, pred_scores = self.model(images, max_len=tokenizer.max_len)
            timing_data['model_inference_time'] += time.time() - t0

            for i, (seqs, scores) in enumerate(zip(pred_seqs, pred_scores)):
                reactions = tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=refs[i]['scale'])
                # Deferred postprocess: parse + deduplicate, no MolScribe/OCR
                pred_reactions = postprocess_reactions_deferred(reactions, image=input_images[idx + i])
                all_reaction_sets.append((idx + i, pred_reactions))

        # Phase 2: Collect ALL molecule bboxes across ALL images → ONE MolScribe call
        if molscribe and self.molscribe is not None:
            has_cache = hasattr(self.molscribe, 'find_cached_smiles') and hasattr(self.molscribe, 'cache_smiles')

            t0 = time.time()
            cache_hits = []  # (list_idx, rxn_idx, bbox_idx, pred)
            all_mol_images = []
            mol_indices = []  # (list_idx, rxn_idx, bbox_idx, figure_id, bbox_coords)

            for list_idx, (img_idx, pred_reactions) in enumerate(all_reaction_sets):
                figure_id = img_idx if has_cache else None
                for rxn_idx, reaction in enumerate(pred_reactions):
                    for bbox_idx, bbox in enumerate(reaction.bboxes):
                        if bbox.is_mol:
                            bbox_coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                            # Try cache lookup
                            if has_cache and figure_id is not None:
                                cached = self.molscribe.find_cached_smiles(figure_id, bbox_coords)
                                if cached is not None:
                                    cache_hits.append((list_idx, rxn_idx, bbox_idx, cached))
                                    continue
                            all_mol_images.append(bbox.image())
                            mol_indices.append((list_idx, rxn_idx, bbox_idx, figure_id, bbox_coords))

            timing_data['modules'].append({
                'name': 'rxn_batched.collect_mol_bboxes',
                'time': time.time() - t0,
                'cache_hits': len(cache_hits),
                'cache_misses': len(all_mol_images)
            })

            # ONE batched MolScribe call for ALL molecules across ALL images
            if all_mol_images:
                t0 = time.time()
                predictions = self.molscribe.predict_images(
                    all_mol_images, return_atoms_bonds=True,
                    batch_size=batch_size, skip_molblock=skip_molblock
                )
                molscribe_time = time.time() - t0
                timing_data['molscribe_time'] = molscribe_time
                timing_data['modules'].append({
                    'name': 'rxn_batched.molscribe.predict_images',
                    'time': molscribe_time,
                    'num_molecules': len(all_mol_images)
                })

                # Distribute results back
                for (list_idx, rxn_idx, bbox_idx, figure_id, bbox_coords), pred in zip(mol_indices, predictions):
                    _, pred_reactions = all_reaction_sets[list_idx]
                    pred_reactions[rxn_idx].bboxes[bbox_idx].set_smiles(
                        pred['smiles'], pred['molfile'], pred['atoms'], pred['bonds']
                    )
                    if has_cache and figure_id is not None:
                        self.molscribe.cache_smiles(figure_id, bbox_coords, pred)

            # Apply cache hits
            if cache_hits:
                for list_idx, rxn_idx, bbox_idx, pred in cache_hits:
                    _, pred_reactions = all_reaction_sets[list_idx]
                    pred_reactions[rxn_idx].bboxes[bbox_idx].set_smiles(
                        pred['smiles'], pred.get('molfile'), pred.get('atoms'), pred.get('bonds')
                    )

        # Phase 3: Collect ALL OCR bboxes across ALL images → ONE threaded OCR batch
        if ocr:
            t0 = time.time()
            ocr_tasks = []  # list of bbox objects
            for _, pred_reactions in all_reaction_sets:
                for reaction in pred_reactions:
                    for bbox in reaction.bboxes:
                        if not bbox.is_mol:
                            ocr_tasks.append(bbox)

            def _ocr_single(bbox):
                return self.ocr_model.readtext(bbox.image(), detail=0)

            if ocr_tasks:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    ocr_results = list(executor.map(_ocr_single, ocr_tasks))
                for bbox, text in zip(ocr_tasks, ocr_results):
                    bbox.set_text(text)

            ocr_time = time.time() - t0
            timing_data['ocr_time'] = ocr_time
            timing_data['ocr_call_count'] = len(ocr_tasks)
            timing_data['modules'].append({
                'name': 'rxn_batched.easyocr.readtext',
                'time': ocr_time,
                'num_calls': len(ocr_tasks)
            })

        # Phase 4: Convert to JSON output
        predictions = []
        for _, pred_reactions in all_reaction_sets:
            predictions.append(pred_reactions.to_json())

        timing_data['total_time'] = time.time() - overall_start
        timing_data['num_images'] = len(input_images)

        if predictions:
            self._last_timing = timing_data

        return predictions

    def get_last_timing(self):
        """Get timing data from the last predict_images call."""
        return getattr(self, '_last_timing', None)

    def predict_image(self, image, **kwargs):
        predictions = self.predict_images([image], **kwargs)
        return predictions[0]

    def predict_image_files(self, image_files: List, **kwargs):
        input_images = []
        for path in image_files:
            image = PIL.Image.open(path).convert("RGB")
            input_images.append(image)
        return self.predict_images(input_images, **kwargs)

    def predict_image_file(self, image_file: str, **kwargs):
        predictions = self.predict_image_files([image_file], **kwargs)
        return predictions[0]

    def draw_predictions(self, predictions, image=None, image_file=None):
        results = []
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        for r in data.pred_reactions:
            fig, ax = plt.subplots(figsize=(w, h))
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            ax.imshow(data.image)
            ax.axis('off')
            r.draw(ax)
            canvas.draw()
            buf = canvas.buffer_rgba()
            results.append(np.asarray(buf))
            plt.close(fig)
        return results

    def draw_predictions_combined(self, predictions, image=None, image_file=None):
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        n = len(data.pred_reactions)
        fig, axes = plt.subplots(n, 1, figsize=(w, h * n))
        if n == 1:
            axes = [axes]
        fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.99))
        canvas = FigureCanvasAgg(fig)
        for i, r in enumerate(data.pred_reactions):
            ax = axes[i]
            ax.imshow(data.image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'reaction # {i}', fontdict={'fontweight': 'bold', 'fontsize': 14})
            r.draw(ax)
        canvas.draw()
        buf = canvas.buffer_rgba()
        result_image = np.asarray(buf)
        plt.close(fig)
        return result_image

class MolDetect:

    def __init__(self, model_path, device = None, coref = False, molscribe = None):
        """
        MolDetect Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        :param molscribe: optional external MolScribe instance (for caching/sharing)
        """
        args = self._get_args()
        if not coref: args.format = 'bbox'
        else: args.format = 'coref'
        states = torch.load(model_path, map_location = torch.device('cpu'))
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.model = self.get_model(args, self.tokenizer, self.device, states['state_dict'])
        self.transform = make_transforms('test', augment=False, debug=False)
        self.ocr_model = self.get_ocr_model()
        self.molscribe = molscribe if molscribe is not None else self.get_molscribe()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')
        # Data
        parser.add_argument('--format', type=str, default='reaction')
        parser.add_argument('--input_size', type=int, default=1333)

        args = parser.parse_args([])
        args.pix2seq = True
        args.pix2seq_ckpt = None
        args.pred_eos = True
        args.is_coco = False
        args.use_hf_transformer = True
        return args


    def get_model(self, args, tokenizer, device, model_states):
        def remove_prefix(state_dict):
            return {k.replace('model.', ''): v for k, v in state_dict.items()}

        model = build_pix2seq_model(args, tokenizer[args.format])
        model.load_state_dict(remove_prefix(model_states), strict=False)
        model.to(device)
        model.eval()
        return model

    def get_molscribe(self):
        ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        molscribe = MolScribe(ckpt_path, device=self.device)
        return molscribe

    def get_ocr_model(self):
        reader = easyocr.Reader(['en'], gpu = (self.device.type == 'cuda'))
        return reader

    def predict_images(self, input_images: List, batch_size = 16, molscribe = False, coref = False, ocr = False, skip_molblock=False):
        import time

        # Reset timing and track overall
        reset_rxnscribe_timing()
        overall_start = time.time()
        timing_data = {
            'modules': [],
            'transform_time': 0,
            'model_inference_time': 0,
            'postprocess_time': 0,
            'molscribe_time': 0,
            'ocr_time': 0,
            'ocr_call_count': 0,
        }

        device = self.device
        if not coref:
            tokenizer = self.tokenizer['bbox']
        else:
            tokenizer = self.tokenizer['coref']

        # Phase 1: Model inference for all images (batched)
        all_raw_bboxes = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]

            # Transform timing
            t0 = time.time()
            images, refs = zip(*[self.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(device)
            timing_data['transform_time'] += time.time() - t0

            # Model inference timing
            t0 = time.time()
            with torch.no_grad():
                pred_seqs, pred_scores = self.model(images, max_len=tokenizer.max_len)
            timing_data['model_inference_time'] += time.time() - t0

            for i, (seqs, scores) in enumerate(zip(pred_seqs, pred_scores)):
                bboxes = tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=refs[i]['scale'])
                all_raw_bboxes.append((idx + i, bboxes))

        # Phase 2: Postprocessing
        if coref:
            # Use batched postprocessing for coref mode (batches MolScribe across ALL images)
            t0 = time.time()
            predictions, batch_timing = self._postprocess_coref_batched(
                input_images,
                all_raw_bboxes,
                molscribe=self.molscribe if molscribe else None,
                ocr=self.ocr_model if ocr else None,
                batch_size=batch_size,
                skip_molblock=skip_molblock
            )
            timing_data['postprocess_time'] = time.time() - t0
            timing_data['molscribe_time'] = batch_timing.get('molscribe_time', 0)
            timing_data['ocr_time'] = batch_timing.get('ocr_time', 0)
            timing_data['ocr_call_count'] = batch_timing.get('ocr_call_count', 0)
            timing_data['modules'].extend(batch_timing.get('modules', []))
        else:
            # Original per-image postprocessing for non-coref mode
            predictions = []
            for img_idx, raw_bboxes in all_raw_bboxes:
                # Set figure context for cache lookup
                if molscribe and self.molscribe is not None:
                    self.molscribe.figure_context = img_idx

                t0 = time.time()
                bboxes = postprocess_bboxes(
                    raw_bboxes,
                    image=input_images[img_idx],
                    molscribe=self.molscribe if molscribe else None,
                    skip_molblock=skip_molblock
                )
                postprocess_elapsed = time.time() - t0
                timing_data['postprocess_time'] += postprocess_elapsed

                # Capture detailed timing from postprocess
                pp_timing = get_rxnscribe_timing()
                for mod in pp_timing.get('modules', []):
                    if 'molscribe' in mod['name']:
                        timing_data['molscribe_time'] += mod['time']
                    timing_data['modules'].append(mod)

                predictions.append(bboxes)

        timing_data['total_time'] = time.time() - overall_start
        timing_data['num_images'] = len(input_images)

        # Store timing for retrieval
        self._last_timing = timing_data

        return predictions

    def get_last_timing(self):
        """Get timing data from the last predict_images call."""
        return getattr(self, '_last_timing', None)

    def _postprocess_coref_batched(self, input_images, all_raw_bboxes, molscribe, ocr, batch_size, skip_molblock):
        """
        Batch MolScribe across all images for coref postprocessing.

        Instead of calling molscribe per-image, we collect ALL molecule bboxes
        from ALL images and run a single batched molscribe.predict_images() call.
        """
        import time
        import cv2
        from .data import ImageData, BBox, record_timing, _get_rxnscribe_timing

        timing_data = {
            'modules': [],
            'molscribe_time': 0,
            'ocr_time': 0,
            'ocr_call_count': 0,
        }

        # Phase 1: Prepare ImageData and BBox objects for all images
        t0 = time.time()
        all_image_data = []
        for img_idx, raw_bboxes in all_raw_bboxes:
            image = input_images[img_idx]
            # Resize image 3x as in original postprocess_coref_results
            image_d = ImageData(image=cv2.resize(np.asarray(image), None, fx=3, fy=3))
            bbox_objects = [BBox(bbox=bbox, image_data=image_d, xyxy=True, normalized=True)
                           for bbox in raw_bboxes['bboxes']]
            corefs = raw_bboxes['corefs']
            all_image_data.append((img_idx, bbox_objects, corefs))
        timing_data['modules'].append({'name': 'coref_batched.prepare_image_data', 'time': time.time() - t0})

        # Phase 2: Collect ALL molecule bboxes across ALL images (with cache checking)
        if molscribe:
            # Check if cache is available
            has_cache = hasattr(molscribe, 'find_cached_smiles') and hasattr(molscribe, 'cache_smiles')

            t0 = time.time()
            cache_hits = []  # List of (list_idx, bbox_idx, cached_pred)
            all_mol_images = []
            mol_indices = []  # List of (list_idx, bbox_idx, img_idx, bbox_coords)
            for list_idx, (img_idx, bbox_objects, corefs) in enumerate(all_image_data):
                for bbox_idx, bbox in enumerate(bbox_objects):
                    if bbox.is_mol:
                        bbox_coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)

                        # Try cache lookup if available (use img_idx as figure_id)
                        if has_cache:
                            cached = molscribe.find_cached_smiles(img_idx, bbox_coords)
                            if cached is not None:
                                cache_hits.append((list_idx, bbox_idx, cached))
                                continue

                        all_mol_images.append(bbox.image())
                        mol_indices.append((list_idx, bbox_idx, img_idx, bbox_coords))
            timing_data['modules'].append({
                'name': 'coref_batched.collect_mol_bboxes',
                'time': time.time() - t0,
                'cache_hits': len(cache_hits),
                'cache_misses': len(all_mol_images)
            })

            # Phase 3: ONE batched MolScribe call for ALL molecules (cache misses only)
            if all_mol_images:
                t0 = time.time()
                predictions = molscribe.predict_images(
                    all_mol_images,
                    return_atoms_bonds=True,
                    batch_size=batch_size,
                    skip_molblock=skip_molblock
                )
                molscribe_time = time.time() - t0
                timing_data['molscribe_time'] = molscribe_time
                timing_data['modules'].append({
                    'name': 'coref_batched.molscribe.predict_images',
                    'time': molscribe_time,
                    'num_molecules': len(all_mol_images)
                })

                # Phase 4: Distribute results back to each image's bboxes and store in cache
                t0 = time.time()
                for (list_idx, bbox_idx, img_idx, bbox_coords), pred in zip(mol_indices, predictions):
                    all_image_data[list_idx][1][bbox_idx].set_smiles(
                        pred['smiles'], pred['molfile'], pred['atoms'], pred['bonds']
                    )
                    # Store in cache
                    if has_cache:
                        molscribe.cache_smiles(img_idx, bbox_coords, pred)
                timing_data['modules'].append({'name': 'coref_batched.distribute_results', 'time': time.time() - t0})

            # Apply cache hits
            if cache_hits:
                t0 = time.time()
                for list_idx, bbox_idx, pred in cache_hits:
                    all_image_data[list_idx][1][bbox_idx].set_smiles(
                        pred['smiles'], pred.get('molfile'), pred.get('atoms'), pred.get('bonds')
                    )
                timing_data['modules'].append({'name': 'coref_batched.apply_cache_hits', 'time': time.time() - t0})

        # Phase 5: OCR for identifiers — parallel with ThreadPoolExecutor
        if ocr:
            t0 = time.time()
            ocr_tasks = []
            for list_idx, (img_idx, bbox_objects, corefs) in enumerate(all_image_data):
                for bbox in bbox_objects:
                    if bbox.is_idt:
                        ocr_tasks.append(bbox)

            def _ocr_single(bbox):
                return ocr.readtext(bbox.image(), detail=0)

            if ocr_tasks:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    ocr_results = list(executor.map(_ocr_single, ocr_tasks))
                for bbox, text in zip(ocr_tasks, ocr_results):
                    bbox.set_text(text)

            ocr_time = time.time() - t0
            timing_data['ocr_time'] = ocr_time
            timing_data['ocr_call_count'] = len(ocr_tasks)
            timing_data['modules'].append({
                'name': 'coref_batched.easyocr.readtext',
                'time': ocr_time,
                'num_calls': len(ocr_tasks)
            })

        # Phase 6: Build final results
        results = []
        for img_idx, bbox_objects, corefs in all_image_data:
            results.append({
                'bboxes': [b.to_json() for b in bbox_objects],
                'corefs': corefs
            })

        return results, timing_data

    def predict_image(self, image, molscribe = False, coref = False, ocr = False, skip_molblock=False):
        predictions = self.predict_images([image], molscribe = molscribe, coref = coref, ocr = ocr, skip_molblock=skip_molblock)
        return predictions[0]

    def predict_image_files(self, image_files: List, batch_size = 16, molscribe = False, coref = False, ocr = False, skip_molblock=False):
        input_images = []
        for path in image_files:
            image = PIL.Image.open(path).convert("RGB")
            input_images.append(image)
        return self.predict_images(input_images, batch_size = batch_size, molscribe = molscribe, coref = coref, ocr = ocr, skip_molblock=skip_molblock)

    def predict_image_file(self, image_file: str, molscribe = False, coref = False, ocr = False, skip_molblock=False, **kwargs):
        predictions = self.predict_image_files([image_file], molscribe = molscribe, coref = coref, ocr = ocr, skip_molblock=skip_molblock)
        return predictions[0]

    def draw_bboxes(self, predictions, image=None, image_file=None, coref = False):
        results = []
        assert image or image_file
        if not coref: data = ImageData(predictions = predictions, image = image, image_file = image_file)
        else: data = CorefImageData(predictions = predictions['bboxes'], image = image, image_file = image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        fig, ax = plt.subplots(figsize = (w, h))
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        ax.imshow(data.image)
        ax.axis('off')
        data.draw_prediction(ax, data.image)
        canvas.draw()
        buf = canvas.buffer_rgba()
        results.append(np.asarray(buf))
        plt.close(fig)
        return results
