# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from decord import VideoReader, cpu, gpu

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import tarfile
from io import BytesIO
import random
import torch
import transformers
import tokenizers
from metamorph.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from metamorph.train.metamorph_trainer import MetaMorphTrainer
from metamorph import conversation as conversation_lib
from metamorph.model import *
from metamorph.mm_utils import tokenizer_image_token
from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_head_type: Optional[str] = field(default='linear')
    image_token_reduction: Optional[str] = field(default='none')
    num_image_tokens: Optional[int] = field(default=256)
    freeze_vision: bool = field(default=False)
    normalize_vision: bool = field(default=False)
    apply_softmax: bool = field(default=False)
    vision_coef: float = field(default=1.0)
    use_vision_ar: bool = field(default=True)
    



@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    vision_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    ############################################################################
    ## MetaMorph Change: Removed fixed template to put Image at the beginning ##
    ############################################################################

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    #############################
    ## End of MetaMorph Change ##
    #############################

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep + conv.roles[1]
    counter = 0
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        # print("rounds are:", rounds)
        re_rounds = [conv.sep.join(rounds[:2])] # system + user + gpt
        # print("then first re_rounds", re_rounds)
        for conv_idx in range(2, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        # print("Final re_rounds:", re_rounds)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        
        target[cur_len:] = IGNORE_INDEX

        counter += 1


        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
    
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    
    #print(sources)
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

   

    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        # print("parts ignored are:", source[0]['value'])
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        # print(tokenized_len)
        target[:tokenized_len] = IGNORE_INDEX
    

    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    # print("My version is", conversation_lib.default_conversation.version, conversation_lib.default_conversation.sep_style)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)



import glob
import os
from PIL import Image


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def process_vstar_question(original_question):
    # Remove the specified parts from the original question
    cleaned_question = original_question.replace('Additional visual information to focus on: ', '').replace('<object>; <object>.\n', '').replace('<object>.', '')
    
    # List of encouragements to imagine bounding boxes
    encouragements = [
        "Ok, let's focus on the important part:",
        "Alright, let's start by visualizing this:",
        "Let's begin by focusing visually on:",
        "Let's explore this visually:",
        "Let's consider this visually:",
        "Visualize the key area:",
        "Let's imagine the critical detail:",
        "Visualize the relevant aspect:",
        "Let's think about this visually:",
        "Here's the visual perspective:",
        "Acknowledging the visual detail:",
        "Alright, let's address this visually:",
        "Visualizing the critical feature:",
        "Acknowledging the visual context:",
        "Starting with the visual aspect:",
    ]
    
    # Choose a random encouragement
    chosen_encouragement = random.choice(encouragements)
    
    # Combine the cleaned question with the encouragement
    new_question = f"{chosen_encouragement} {cleaned_question}"
    
    return new_question


def process_vstar_answer(original_answer, target_instances):
    # Extract only the names of the objects from target instances
    object_names = [f"'{instance['name']}'" for instance in target_instances]
    # objects_description = ", ".join(object_names)
    
    if target_instances is not None:
        # Create the new answer structure
        new_answer = "I will identify the key visual elements and answer the question. The key elements are "
        
        elements = [f"{entry['name']}" for entry in target_instances]
        
        if len(elements) > 1:
            new_answer += ", ".join(elements[:-1]) + ", and " + elements[-1] + "."
        elif len(elements) == 1:
            new_answer += elements[0] + "."
        else:
            new_answer = new_answer.rstrip() + "."  # Remove trailing space if no elements


    # Create the new answer structure
    new_answer += f" Then I will identify these elements with bounding boxes <image>. Based on these highlighted areas, here's my response to the question: {original_answer}"
    
    return new_answer


from PIL import Image, ImageDraw

def draw_bounding_boxes(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = bbox['bbox']
        draw.rectangle([x, y, x+w, y+h], outline="red", width=5)
        draw.text((x, y-20), bbox['name'], fill="red")
    return image

def extract_segmented_regions(image, bboxes):
    segmented_regions = []
    for bbox in bboxes:
        x, y, w, h = bbox['bbox']
        region = image.crop((x, y, x+w, y+h))
        segmented_regions.append({
            'name': bbox['name'],
            'image': region
        })
    return segmented_regions


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        

        if "metacliptar" in data_path:
            self.use_metaclip = True
            self.add_prompts = True
        elif "metaclip" in data_path:
            self.use_metaclip = False
            self.add_prompts = True
        else:
            self.use_metaclip = False
            self.add_prompts = False

        
        self.data_args = data_args




        ############################################################################
        ## MetaMorph Change: We design different prefix for Visual Reasoning Task ##
        ############################################################################

        # Optional prefixes for human responses (encouraging visual imagination)
        self.cot_human_prefixes = [
            "Imagine the important part visually:",
            "Visualize the critical area,",
            "Picture the relevant detail:",
            "Consider the visual aspect,",
            "Focus on the key visual element:",
            "Imagine the scene with this focus,",
            "Reflect on the visual details:",
            "Visualize the context,",
            "Visualize the significant aspect,",
            "Think about this visually,",
            "Visualize the essential detail:",
            "Envision the visual perspective:",
            ""
        ]

        # Optional prefixes for GPT responses (acknowledging and focusing visually)
        self.cot_gpt_prefixes = [
            "Ok, let's focus on the important part:",
            "Alright, let's start by visualizing this:",
            "Let's begin by focusing visually on:",
            "Let's explore this visually:",
            "Let's consider this visually:",
            "Visualize the key area:",
            "Let's imagine the critical detail:",
            "Visualize the relevant aspect:",
            "Let's think about this visually:",
            "Here's the visual perspective:",
            "Acknowledging the visual detail:",
            "Alright, let's address this visually:",
            "Visualizing the critical feature:",
            "Acknowledging the visual context:",
            "Starting with the visual aspect:",
            ""
        ]

        #############################
        ## End of MetaMorph Change ##
        #############################


        self.data_path = data_path

        self.line_offsets = self._index_file()
        self.length = len(self.line_offsets)


        rank0_print("Formatting non MetaCLIP ...Skip in lazy mode")
        self.tokenizer = tokenizer

    
    
    def _index_file(self) -> List[int]:
        """Creates an index of byte offsets for each line in the file."""
        offsets = [0]
        with open(self.data_path, 'rb') as file:
            while file.readline():
                offsets.append(file.tell())
        return offsets[:-1]  # Last offset is EOF

    def _get_length(self):
        """Returns the number of samples in the dataset."""
        return len(self.line_offsets)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.length
    
    
    ############################################################################
    ## MetaMorph Change: We design different prefix for Video Prediction Task ##
    ############################################################################

    def generate_text_description(self, frames, interval_t):
        image_tags = ['<image>' for _ in frames]
        if len(frames) == 2:  # Special case for two frames
            variants = [
                (
                    f"{image_tags[0]} Predict visually what's going to happen in {interval_t} seconds.",
                    f"{image_tags[1]}"
                ),
                (
                    f"Check out this snapshot {image_tags[0]}. Fast forward {interval_t} seconds - what do you think we'll see?",
                    f"Time travel complete! Here's your future frame: {image_tags[1]}"
                ),
                (
                    f"{image_tags[0]} Imagine this scene evolving over the next {interval_t} seconds. What's your visual prediction?",
                    f"Ta-da! Your crystal ball reveals: {image_tags[1]}"
                ),
                (
                    f"Given this starting point {image_tags[0]}, can you forecast the visual outcome {interval_t} seconds later?",
                    f"Forecast delivered! Your updated scene: {image_tags[1]}"
                ),
                (
                    f"{image_tags[0]} If we hit the fast-forward button for {interval_t} seconds, what would we likely observe?",
                    f"Fast-forward complete! Observe the result: {image_tags[1]}"
                ),
                (
                    f"Picture this: {image_tags[0]}. Now, let's jump {interval_t} seconds into the future. What do you envision?",
                    f"Vision materialized! Here's your future snapshot: {image_tags[1]}"
                ),
                (
                    f"{image_tags[0]} Using your imagination, how do you think this scene will transform in {interval_t} seconds?",
                    f"Transformation complete! Behold the new scene: {image_tags[1]}"
                ),
                (
                    f"Starting from this image {image_tags[0]}, predict the visual progression after {interval_t} seconds.",
                    f"Progression predicted! Your updated image awaits: {image_tags[1]}"
                ),
                (
                    f"{image_tags[0]} If we could peek {interval_t} seconds into the future, what would this scene look like?",
                    f"Peek granted! Future scene revealed: {image_tags[1]}"
                ),
                (
                    f"Analyze this initial frame {image_tags[0]}. How do you expect it to change in {interval_t} seconds?",
                    f"Analysis complete! Expected change visualized: {image_tags[1]}"
                ),
                (
                    f"{image_tags[0]} Let's play futurist. Project this scene forward by {interval_t} seconds.",
                    f"Future projected! Your advanced timeline shows: {image_tags[1]}"
                ),
                (
                    f"{image_tags[0]} Please predict the next state:",
                    f"This is my answer: {image_tags[1]}"
                )
            ]

            # Needed code
            question, answer = random.choice(variants)
            descriptions = (question, answer, None)
        else:

            def generate_descriptions(image_tags, interval_t):
                descriptions = []
                num_frames = len(image_tags)
                
                # Randomly select start and end frames
                start_frame = random.randint(0, num_frames - 2)
                end_frame = random.randint(start_frame + 1, num_frames - 1)
                selected_frames = image_tags[start_frame:end_frame + 1]
                selected_num_frames = len(selected_frames)
                total_time = (selected_num_frames - 1) * interval_t

                # Simple prediction
                descriptions.append((
                    f"I have a video starting with this frame: {selected_frames[0]}. Can you predict what happens in the next {selected_num_frames - 1} frames, each {interval_t} seconds apart?",
                    f"Certainly! I'll generate the next {selected_num_frames - 1} frames, each {interval_t} seconds apart. Here are my predictions: {' '.join(selected_frames[1:])}",
                    lambda images: images[start_frame:end_frame + 1]
                ))

                # Split prediction
                split_index = random.randint(1, selected_num_frames - 1)
                descriptions.append((
                    f"I've seen the first {split_index} frame{'s' if split_index > 1 else ''} of a {selected_num_frames}-frame video sequence, each frame {interval_t} seconds apart: {''.join(selected_frames[:split_index])}. What do you think happens in the remaining {selected_num_frames - split_index} frame{'s' if selected_num_frames - split_index > 1 else ''}?",
                    f"Based on the first {split_index} frame{'s' if split_index > 1 else ''} you've seen, I'll predict the remaining {selected_num_frames - split_index} frame{'s' if selected_num_frames - split_index > 1 else ''}, each {interval_t} seconds apart. Here's what I think happens next: {''.join(selected_frames[split_index:])}",
                    lambda images: images[start_frame:end_frame + 1]
                ))

                # Reverse prediction
                descriptions.append((
                    f"I have the final frame of a {selected_num_frames}-frame video, where each frame is {interval_t} seconds apart: {selected_frames[-1]}. Can you work backwards and predict the previous {selected_num_frames - 1} frames?",
                    f"Interesting challenge! I'll work backwards to generate the previous {selected_num_frames - 1} frames, each {interval_t} seconds apart. Here's my reverse prediction: {''.join(selected_frames[-2::-1])}",
                    lambda images: images[start_frame:end_frame + 1][::-1]
                ))

                # Alternating frames
                descriptions.append((
                    f"I have a {selected_num_frames}-frame video where I can only see every other frame, each {interval_t} seconds apart: {' '.join(selected_frames[::2])}. Can you predict the {len(selected_frames[1::2])} missing frames?",
                    f"Of course! I'll predict the {len(selected_frames[1::2])} missing frames to complete the {selected_num_frames}-frame sequence, where each frame is {interval_t} seconds apart. Here are the frames I think are missing: {''.join(selected_frames[1::2])}",
                    lambda images: images[start_frame:end_frame + 1][::2] + images[start_frame:end_frame + 1][1::2]
                ))

                # Rearrange frames
                shuffled_indices = list(range(selected_num_frames))
                random.shuffle(shuffled_indices)
                shuffled_frames = [selected_frames[i] for i in shuffled_indices]
                descriptions.append((
                    f"I have a series of {selected_num_frames} video frames, each {interval_t} seconds apart, but they're all mixed up: {' '.join(shuffled_frames)}. Can you put them in the right order?",
                    f"I'd be happy to help! You gave me these {selected_num_frames} frames, each {interval_t} seconds apart, in this order: {' '.join(shuffled_frames)}.",
                    lambda images: [images[start_frame:end_frame + 1][i] for i in shuffled_indices] + images[start_frame:end_frame + 1]
                ))

                # Determine time interval
                descriptions.append((
                    f"I have a series of {selected_num_frames} video frames: {' '.join(selected_frames)}. Can you tell me how much time passes between each frame?",
                    f"To determine the time interval, I'll analyze the changes between each of the {selected_num_frames} frames. Based on my analysis, I estimate that approximately {interval_t} seconds pass between each frame in this {selected_num_frames}-frame sequence.",
                    lambda images: images[start_frame:end_frame + 1]
                ))

                # New case: Predict next X images
                predict_count = random.randint(1, min(3, num_frames - end_frame))  # Predict 1 to 3 frames, or up to the end of the sequence
                descriptions.append((
                    f"Here is a sequence of images: {' '.join(selected_frames)}. Predict the next {predict_count} image{'s' if predict_count > 1 else ''} in the sequence.",
                    f"Based on the given sequence of images, I'll predict the next {predict_count} image{'s' if predict_count > 1 else ''} in the sequence. Here {'are' if predict_count > 1 else 'is'} my prediction{'s' if predict_count > 1 else ''}: {' '.join(image_tags[end_frame + 1:end_frame + 1 + predict_count])}",
                    lambda images: images[start_frame:end_frame + 1 + predict_count]
                ))

                return descriptions
    
            descriptions_candidate = generate_descriptions(image_tags, interval_t)
            descriptions = random.choice(descriptions_candidate)

        return descriptions
    #############################
    ## End of MetaMorph Change ##
    #############################


    def processVideo(self, video_path, cutoff=45, interval_t=None):
        device_ctx = cpu(0)

        vr = VideoReader(video_path, num_threads=-1, ctx=device_ctx)
        
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        video_length = num_frames / fps if fps > 0 else 0  # Total video length in seconds
        
        if video_length == 0:
            raise ValueError("Unable to determine video length. The video might be corrupted or empty.")

        if interval_t is None:
            min_interval = max(2 / fps, 0.1)  # Minimum of 2 frames or 0.1 seconds
            max_interval = min(10, max(video_length / 2, min_interval))
            interval_t = round(random.uniform(min_interval, max_interval), 1)  # Round to 1 decimal place
        else:
            interval_t = round(interval_t, 1)  # Ensure provided interval_t is also rounded

        frames = []
        current_time = 0
        while current_time < video_length and len(frames) < cutoff:
            frame_index = min(int(current_time * fps), num_frames - 1)
            frames.append(vr[frame_index].asnumpy())
            current_time += interval_t

        rgb_frames = [frame[:, :, :3] if frame.shape[2] == 4 else frame for frame in frames]

        return rgb_frames, interval_t


    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        try:
            with open(self.data_path, 'r') as file:
                file.seek(self.line_offsets[i])
                sources = json.loads(file.readline().strip())
                
            dat = sources

            if isinstance(i, int):
                sources = [sources]
            processed_images = []

            #########################################################################
            ## MetaMorph Change: We preprocess different types of input data here: ##
            ##  1. Pure Video Code, we designed many templates to transform videos ##
            ##  2. Video QA, we process videos 1FPS                                ##
            ##  3. Image Transformation, predict transformed images                ##
            ##  4. Visual Reasoning, predict intermediate **visual** tokens        ##
            #########################################################################
      

            if 'image' in sources[0] and sources[0]['image'] is not None:
                processor = self.data_args.image_processor
                if self.use_metaclip:
                    images = sources[0]['image']
                else:
                    
                    image_file = dat['image']

                    if type(image_file) is list:
                        if "visual_cot" in image_file[0]:
                            # Training on Visual CoT here
                            image_path = image_file[0]
                            bounding_box_str = image_file[1]

                                # Extract bounding box coordinates from the string
                            box_coords = bounding_box_str.split('###')[1].strip('[]')
                            box_coords = [int(coord.strip()) for coord in box_coords.split(',')]

                            # Load the image
                            image = Image.open(image_path)

                            images = [image.convert('RGB')]

                            # Extract the bounding box region
                            bbox_image = image.crop(box_coords)

                            # Convert bounding box image to RGB
                            bbox_image_rgb = bbox_image.convert('RGB')

                            # Append RGB bounding box image to list
                            images.append(bbox_image_rgb)

                            human_response = sources[0]['conversations'][0]["value"]
                            gpt_response = sources[0]['conversations'][3]["value"]

                            # Clean up human and GPT responses
                            human_response = human_response.split('Please provide the bounding box coordinate of')[0].strip()
                            # Randomly select from self.cot_human_prefixes and self.cot_gpt_prefixes
                            human_prefix = random.choice(self.cot_human_prefixes)
                            gpt_prefix = random.choice(self.cot_gpt_prefixes)

                            # Update human_response and gpt_response with the randomly chosen prefixes
                            human_response = f"{human_prefix} {human_response}"
                            gpt_response = f"{gpt_prefix} <image> {gpt_response}"
                            
                            sources[0]["conversations"] = [
                                {
                                    "from": "human",
                                    "value": human_response
                                },
                                {
                                    "from": "gpt",
                                    "value": gpt_response
                                }
                            ]

                        else:
                            # This case just other already processed multi image files:
                            images = [Image.open(image).convert('RGB') for image in image_file]

                    elif any(ext in image_file for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']):
                        if "k700" in sources[0].get('id', 'NA'):

                            images, interval = self.processVideo(image_file, interval_t=1)

                            image_tags = ''.join(['<image>' for _ in images])
                            sources[0]["conversations"][0]["value"]  = image_tags + sources[0]["conversations"][0]["value"]

                        else:
                            images, interval = self.processVideo(image_file)

                            if (len(images)<2):
                                print("error here!", images, interval, image_file)

                            human_response, gpt_response, reorder_func = self.generate_text_description(images, interval)
                            if reorder_func is not None:
                                images = reorder_func(images)

                            sources[0]["conversations"] = [
                                    {
                                        "from": "human",
                                        "value": human_response
                                    },
                                    {
                                        "from": "gpt",
                                        "value": gpt_response
                                    }
                                ]
                    elif "vstar" in str(sources[0].get('id', 'NA')):
                            entry = sources[0]
        
                            # Process the question
                            human_response = process_vstar_question(entry['conversations'][0]['value'])
                            
                            # Process the answer
                            gpt_response = process_vstar_answer(entry['conversations'][1]['value'], entry['target_instances'])
                            
                            # Update the conversations in the entry
                            sources[0]["conversations"] = [
                                {
                                    "from": "human",
                                    "value": human_response
                                },
                                {
                                    "from": "gpt",
                                    "value": gpt_response
                                }
                            ]

                            
                            # Process the image
                            original_image = Image.open(entry['image']).convert('RGB')
                            boxed_image = draw_bounding_boxes(original_image.copy(), entry['target_instances'])


                            # Create the images list
                            images = [original_image]
                            images.append(boxed_image)

                    else:
                        images = Image.open(image_file).convert('RGB')

                ###############################################################
                ## End of MetaMorph Change: Processed different data formats ##
                ###############################################################
                
                if not isinstance(images, list):
                    images = [images]

                processed_images = []
                for image in images:

                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    
                    processed_images.append(image)

                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            
            has_image = (self.use_metaclip or ('image' in dat and (dat['image'] is not None)))

            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=has_image)
            
            count = torch.sum(data_dict["input_ids"].eq(-200)).item()
            if count!=len(processed_images):
                print("Bugbugbug!", count, len(processed_images), sources)
                return self.__getitem__((i + 1) % self.__len__())

            if count>70:
                return self.__getitem__((i + 1) % self.__len__())

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])

            # image exist in the data
            if has_image:
                data_dict['image'] = processed_images
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]

            return data_dict
        
        except Exception as e:
            print(f"Bug at {i}: {str(e)}")
            return self.__getitem__((i + 1) % self.__len__())
            


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # Flatten the list of lists into a single list of tensors
            flat_list = [item for sublist in images for item in sublist]
            # Stack all the tensors in the list into a single tensor
            stacked_tensor = torch.stack(flat_list)
            batch['images'] = stacked_tensor

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


import numpy as np

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




###############################################################################
## MetaMorph Change: Add W&B Recall to also record Image Autoregressive Loss ##
###############################################################################

from transformers.integrations.integration_utils import WandbCallback

def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v

            # Log the loss_image_ar if available
            if hasattr(model, "loss_language") and model.loss_language is not None:
                logs["loss_language"] = model.loss_language
                
            if hasattr(model, "loss_image_ar") and model.loss_image_ar is not None:
                logs["loss_image_ar"] = model.loss_image_ar

            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}

            non_scalar_logs = rewrite_logs(non_scalar_logs)

            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})

######################################################################################
## End of MetaMorph Change: Add W&B Recall to also record Image Autoregressive Loss ##
######################################################################################

def train(attn_implementation=None):
    global local_rank

    seed = 42
    set_seed(42)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else None))
    print("compute_dtype is", compute_dtype)


    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))


    if model_args.vision_tower is not None:

        model = MetaMorphLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
            use_vision_ar=model_args.use_vision_ar, 
            vision_coef=model_args.vision_coef,
            vision_head=model_args.vision_head_type,
            normalize_vision=model_args.normalize_vision,
            apply_softmax=model_args.apply_softmax,

            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.to(dtype=compute_dtype)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False) 

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    if "Qwen"in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token = "<|endoftext|>"

    if "Llama-3"in model_args.model_name_or_path:
        # tokenizer.pad_token = tokenizer.unk_token = "<|reserved_special_token_250|>"
        tokenizer.pad_token = tokenizer.unk_token = "<|end_of_text|>"

    if model_args.vision_tower is not None and (model_args.vision_tower != "None"):
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_lr = training_args.vision_lr

        model.config.vision_head_type = model_args.vision_head_type
        model.config.image_token_reduction = model_args.image_token_reduction
        model.config.num_image_tokens = model_args.num_image_tokens
        model.config.freeze_vision = model_args.freeze_vision
        model.config.normalize_vision = model_args.normalize_vision
        model.config.apply_softmax = model_args.apply_softmax

        model.config.vision_coef = model_args.vision_coef
        model.config.use_vision_ar = model_args.use_vision_ar

        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)


        if model_args.freeze_vision:
            model.model.get_vision_tower().requires_grad_(False) 

            if model_args.mm_projector_type == "mlpsoftmax":
                print("Freeze first linear!!!")

                # Also freeze the first linear layer
                projector = model.get_model().mm_projector

                # Freeze the first linear layer
                first_linear_layer = projector[0]  # Access the first Linear layer

                # Set requires_grad to False for all parameters in the first linear layer
                for param in first_linear_layer.parameters():
                    param.requires_grad = False

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if "wandb" in training_args.report_to:
        # rm wandb from training_args.report_to so it doesn't get passed to the Trainer
        training_args.report_to.remove("wandb")
        assert "wandb" not in training_args.report_to, training_args.report_to

    trainer = MetaMorphTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[CustomWandbCallback()],
                    **data_module)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
