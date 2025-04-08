# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Optional, Tuple, Union

from transformers import LogitsProcessor

import torch
import torch.nn as nn

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty=1.1):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores

from transformers.cache_utils import Cache

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import wandb

import torch.nn.functional as F

def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)
    return loss

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..metamorph_arch import MetaMorphMetaModel, MetaMorphMetaForCausalLM

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

_CONFIG_FOR_DOC = "LlamaConfig"

class MetaMorphConfig(LlamaConfig):
    model_type = "metamorph_llama"


class MetaMorphLlamaModel(MetaMorphMetaModel, LlamaModel):
    config_class = MetaMorphConfig

    def __init__(self, config: LlamaConfig, vision_delay_load=True):
        super(MetaMorphLlamaModel, self).__init__(config, vision_delay_load=vision_delay_load)


def infonce_loss(inputs_embeds_images_detached, predicted_embeds_images, temperature=0.03):
    # Ensure the input tensors have the same shape
    assert inputs_embeds_images_detached.shape == predicted_embeds_images.shape
    
    # Get the batch size (n) and embedding dimension (d)
    n, d = inputs_embeds_images_detached.shape
    
    # Normalize the embeddings to unit vectors
    inputs_embeds_images_detached = F.normalize(inputs_embeds_images_detached, p=2, dim=1)
    predicted_embeds_images = F.normalize(predicted_embeds_images, p=2, dim=1)
    
    # Compute similarity scores
    similarity_matrix = torch.mm(predicted_embeds_images, inputs_embeds_images_detached.T) / temperature
    
    # Create labels for the positive pairs
    labels = torch.arange(n).to(similarity_matrix.device)
    
    # Compute the cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def print_non_zero_token_ranges(tensor):
    """
    Print the ranges of non-zero tokens for each sample in the tensor.

    Parameters:
    tensor (torch.Tensor): A tensor of shape (batch_size, num_tokens, dim).

    Returns:
    None
    """
    batch_size, num_tokens, dim = tensor.shape
    for i in range(batch_size):
        sample = tensor[i]
        non_zero_indices = (sample != 0).any(dim=1).nonzero(as_tuple=True)[0]
        
        if len(non_zero_indices) > 0:
            ranges = []
            start = non_zero_indices[0].item()
            end = start
            
            for idx in non_zero_indices[1:]:
                idx = idx.item()
                if idx == end + 1:
                    end = idx
                else:
                    ranges.append((start, end))
                    start = idx
                    end = idx
            
            ranges.append((start, end))
            
            ranges_str = ', '.join([f"{start}-{end}" for start, end in ranges])
            print(f"Sample {i}: Non-zero token ranges {ranges_str}")
        else:
            print(f"Sample {i}: All tokens are zero")

def l1_loss_fn(z, h, loss_exp=1):
    
    # loss = 0.
    # # Compute loss and accumulate for each mask-enc/mask-pred pair
    # for zi, hi in zip(z, h):
    #     loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
    # loss /= len(z)

    loss = torch.abs(z - h).mean()

    return loss

def mse_loss_fn(z, h, loss_exp=1):
    
    loss = 0.
    # Compute loss and accumulate for each mask-enc/mask-pred pair
    for zi, hi in zip(z, h):
        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
    loss /= len(z)

    return loss

import torch.nn.functional as F

class MetaMorphLlamaForCausalLM(LlamaForCausalLM, MetaMorphMetaForCausalLM):
    config_class = MetaMorphConfig

    def __init__(self, config, use_vision_ar=True, vision_head = "None", vision_coef=1.0, normalize_vision = False, apply_softmax = False, vision_delay_load=True, full_ar=False):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MetaMorphLlamaModel(config, vision_delay_load=vision_delay_load)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.normalize_vision = normalize_vision
        self.apply_softmax = apply_softmax

        try:
            if config.normalize_vision:
                self.normalize_vision = True
        except:
            pass
        try:
            vision_head = config.vision_head_type
        except:
            pass
        
        if vision_head == "linear":
            self.vision_head = nn.Linear(config.hidden_size, config.hidden_size)
        elif vision_head == "mlp": 

            print("It is a mlp vision head")

            self.vision_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1152),
            )
        elif vision_head == "mlp2x_gelu": 

            print("It is a mlp2x vision head")

            self.vision_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1152),
            )
        else:
             self.vision_head = nn.Linear(config.hidden_size, 1152)

        self.use_vision_ar = use_vision_ar

        self.vision_coef = vision_coef
        


        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
        decoding = False, 
        image_features = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if image_positions is not None:
            # Assuming new_input_embeds is already padded and has the same shape as new_image_position_padded
            # Expand new_image_position_padded to match the last dimension of new_input_embeds
            expanded_image_position = image_positions.unsqueeze(-1).contiguous().expand_as(inputs_embeds).contiguous()
            # Convert expanded_image_position to the same dtype as new_input_embeds for multiplication
            expanded_image_position = expanded_image_position.to(dtype=inputs_embeds.dtype).contiguous()
            # Element-wise multiplication to zero out positions where image position is zero
            inputs_embeds_images = (inputs_embeds * expanded_image_position).contiguous()
            inputs_embeds_images = (inputs_embeds_images[:, 1:, :]).contiguous()


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if decoding:

            pred_z = hidden_states[:, -1, :]

            pred_z = self.vision_head(pred_z)

            if self.normalize_vision:
                pred_z = F.normalize(pred_z, p=2, dim=-1)

            if self.apply_softmax:
                pred_z = F.softmax(pred_z / 0.07, dim=-1)

            prediction = self.get_model().mm_projector(pred_z)
            
            hidden_states[:, -1, :] = prediction



        #######################################
        ## Get the input embed for AR Output ##
        #######################################
        if image_positions is not None:
            # Slice image_positions and adjust dimensions to match inputs_embeds
            expanded_image_position = image_positions[:, 1:].unsqueeze(-1).contiguous().expand_as(inputs_embeds_images).contiguous()
            # Convert expanded_image_position to the same dtype as new_input_embeds for multiplication
            expanded_image_position = expanded_image_position.to(dtype=inputs_embeds.dtype).contiguous()
            # Element-wise multiplication to zero out positions where image position is zero
            predicted_embeds_images = (hidden_states[:, :-1, :] * expanded_image_position).contiguous()


        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            # print(shift_labels)
            loss = loss_fct(shift_logits, shift_labels)



            ##########################
            # Compute Image AR Loss ##
            ##########################
            if image_positions is not None:

                if image_features is not None:

                    # Create a mask to identify non-zero entries across the entire batch
                    non_zero_mask = expanded_image_position.view(-1, self.config.hidden_size) != 0

                    # Filter out non-zero entries from predicted_embeds_images
                    predicted_embeds_images = predicted_embeds_images.view(-1, self.config.hidden_size)
                    non_zero_predicted_embeds_images = predicted_embeds_images[non_zero_mask]

                    # Reshape the filtered tensor to the desired shape (number of non-zero tokens across all batches, self.config.hidden_size)
                    pred = non_zero_predicted_embeds_images.view(-1, self.config.hidden_size)
                    pred = self.vision_head(pred)
                    if self.normalize_vision:
                        pred = F.normalize(pred, p=2, dim=-1)

                    if self.apply_softmax:
                        pred = F.softmax(pred / 0.07, dim=-1)

                    target_image_features = image_features.contiguous().view(-1, image_features.size(-1))
                    
                    if self.apply_softmax:
                        # Small constant to prevent log(0)
                        epsilon = 1e-10

                        # Compute cross-entropy loss
                        loss_image_ar = -(target_image_features * torch.log(pred + epsilon)).sum(dim=1).mean()

                    elif self.normalize_vision:
                        # Use cosine similarity loss
                        try:
                            cosine_similarity = F.cosine_similarity(target_image_features, pred, dim=-1)
                            loss_image_ar =  -cosine_similarity.mean()  # Convert similarity to loss
                        except:
                            loss_image_ar = loss
                
                    else:
                        # Use L1 loss as before
                        loss_image_ar = mse_loss_fn(target_image_features, pred)

                else:
                    loss_image_ar = loss

                self.loss_language = loss.item()

                self.loss_image_ar = loss_image_ar.item()
                


                if self.use_vision_ar:
                    
                    if loss_image_ar.item() != 0:

                        loss += self.vision_coef * loss_image_ar
                

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        if decoding:

            return CausalLMOutputWithPast(
                loss=pred_z,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
        
        

    def greedy_decode(self, position_ids, attention_mask, inputs_embeds, start_image_token_id=128256, end_image_token_id=128257, eos_token_id=[128001,128009], do_sample=None, temperature=None, top_p=None, num_beams=None, max_new_tokens=1024, use_cache=None, output_image=False):

        past_key_values = None
        in_image_mode = False
        generated_ids_list = []
        total_image_tokens = 0
        total_output_tokens = 0
        
        use_cache = False
        
        image_embeds_list = []


        # Initialize attention_mask if it's None
        if attention_mask is None:
            _, num_tokens, _ = inputs_embeds.shape
            attention_mask = torch.ones((1, num_tokens), dtype=torch.long, device=inputs_embeds.device)

        num_image_tokens = self.get_model().vision_tower.image_token_len

        while True:
           
            attention_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

            outputs = self.llm_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                decoding=in_image_mode, 
            )

            image_embed = outputs.loss

            next_token_logits = outputs.logits[:, -1, :]
            next_embed = outputs.hidden_states[:, -1, :].unsqueeze(0)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            next_token_embed = self.model.embed_tokens(next_token)
           

            if (not in_image_mode) and next_token.item() == start_image_token_id:
                print("Enter image mode")
                in_image_mode = True
                generated_ids_list.append(next_token.item())
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)


            elif (in_image_mode) and (total_image_tokens<num_image_tokens):

                total_image_tokens += 1

                image_embeds_list.append(image_embed)

                inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)

                if total_image_tokens==num_image_tokens:
                    in_image_mode = False

            elif next_token.item() == end_image_token_id:
                in_image_mode = False
                total_image_tokens = 0
                generated_ids_list.append(next_token.item())
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
            
            else:
                # Append token embeddings
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                generated_ids_list.append(next_token.item())

            total_output_tokens += 1

            if next_token.item() in eos_token_id:
                break

            if total_output_tokens > max_new_tokens:
                break

            past_key_values = outputs.past_key_values
          
        if image_embeds_list:
            image_embeds_tensor = torch.cat(image_embeds_list, dim=0)
        else:
            image_embeds_tensor = torch.tensor([], dtype=torch.float32, device=inputs_embeds.device)

        output = [torch.tensor(generated_ids_list, dtype=torch.int32, device=inputs_embeds.device)]


         # Perform random cropping and compute cosine similarity
        if output_image:
            return output, image_embeds_tensor
        return output

        

        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[List[torch.FloatTensor]] = None,
        image_embeds = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        image_positions = None

        target_prob = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels, 
                image_positions,
                target_prob,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                image_embeds
            )
       

        return self.llm_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_positions=image_positions,
            image_features = target_prob,
        )


      

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        output_image = False, 
        use_customize_greedy = True,
        image_embeds = None, 
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
       
        if images is not None or image_embeds is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _, 
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                image_embeds=image_embeds,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
     
        
        if use_customize_greedy:
            return self.greedy_decode(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_image = output_image, 
                **kwargs
            )
        else:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # print("input id and input_embeds shape", input_ids, inputs_embeds.shape)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # print("prepared:", inputs.keys())
        # print("shape:", inputs["inputs_embeds"].shape)
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("metamorph_llama", MetaMorphConfig)
AutoModelForCausalLM.register(MetaMorphConfig, MetaMorphLlamaForCausalLM)
