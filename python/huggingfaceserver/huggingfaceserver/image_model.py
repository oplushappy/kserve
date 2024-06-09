# Copyright 2024 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import pathlib
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from kserve.errors import InferenceError
from kserve.logging import logger
from kserve.model import PredictorConfig
from kserve.protocol.infer_type import InferInput, InferRequest, InferResponse
from kserve.utils.utils import from_np_dtype, get_predict_input, get_predict_response
from PIL import Image
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    TensorType,
)

from kserve import Model

from .task import (
    MLTask,
    get_model_class_for_task,
    infer_task_from_model_architecture,
    is_generative_task,
    is_image_task,
)

PILImage = Image.Image


class HuggingfaceImageModel(Model):  # pylint:disable=c-extension-no-member
    task: MLTask
    model_config: PretrainedConfig
    model_id_or_path: Union[pathlib.Path, str]
    tensor_input_names: Optional[str]
    model_revision: Optional[str]
    trust_remote_code: bool
    ready: bool = False
    image_processor_revision: Optional[str]
    _model: Optional[PreTrainedModel] = None
    _device: torch.device

    def __init__(
        self,
        model_name: str,
        model_id_or_path: Union[pathlib.Path, str],
        model_config: Optional[PretrainedConfig] = None,
        task: Optional[MLTask] = None,
        dtype: torch.dtype = torch.float32,
        tensor_input_names: Optional[str] = None,
        model_revision: Optional[str] = None,
        image_processor_revision: Optional[str] = None,
        trust_remote_code: bool = False,
        predictor_config: Optional[PredictorConfig] = None,
    ):
        super().__init__(model_name, predictor_config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id_or_path = model_id_or_path
        self.dtype = dtype
        self.tensor_input_names = tensor_input_names
        self.model_revision = model_revision
        self.image_processor_revision = image_processor_revision
        self.trust_remote_code = trust_remote_code

        if model_config:
            self.model_config = model_config
        else:
            self.model_config = AutoConfig.from_pretrained(self.model_id_or_path)

        if task:
            self.task = task
            try:
                inferred_task = infer_task_from_model_architecture(self.model_config)
            except ValueError:
                inferred_task = None
            if inferred_task is not None and inferred_task != task:
                logger.warn(
                    f"Inferred task is '{inferred_task.name}' but"
                    f" task is explicitly set to '{self.task.name}'"
                )
        else:
            self.task = infer_task_from_model_architecture(self.model_config)

        if is_generative_task(self.task):
            raise RuntimeError(
                f"Image Encoder model does not support generative task: {self.task.name}"
            )
        if not is_image_task(self.task):
            raise RuntimeError(
                f"Image Encoder model does not support text encoder task: {self.task.name}"
            )

    def load(self) -> bool:
        model_id_or_path = self.model_id_or_path

        # device_map = "auto" enables model parallelism but all model architcture dont support it.
        # For pre-check we initialize the model class without weights to check the `_no_split_modules`
        # device_map = "auto" for models that support this else set to either cuda/cpu
        with init_empty_weights():
            self._model = AutoModel.from_config(self.model_config)

        device_map: str = str(self._device)

        if self._model._no_split_modules:
            device_map = "auto"

        image_processor_kwargs = {}
        model_kwargs = {}

        if self.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            image_processor_kwargs["trust_remote_code"] = True

        model_kwargs["torch_dtype"] = self.dtype

        # load hugging face image preprocessor
        self._image_processor = AutoImageProcessor.from_pretrained(
            str(model_id_or_path),
            revision=self.image_processor_revision,
            **image_processor_kwargs,
        )
        logger.info("Successfully loaded image processor")

        # load huggingface model using from_pretrained for inference mode
        if not self.predictor_host:
            model_cls = get_model_class_for_task(self.task)
            self._model = model_cls.from_pretrained(
                model_id_or_path,
                revision=self.model_revision,
                device_map=device_map,
                **model_kwargs,
            )
            self._model.eval()
            self._model.to(self._device)
            logger.info(
                f"Successfully loaded huggingface model from path {model_id_or_path}"
            )
        self.ready = True
        return self.ready

    def preprocess(
        self,
        payload: Union[Dict, InferRequest],
        context: Dict[str, Any],
    ) -> Union[Dict, InferRequest]:
        # Get the images from the payload
        instances = get_predict_input(payload)
        # Serialize to tensor
        if self.predictor_host:
            # still can add
            inputs = self._image_processor(
                instances[0],  # image
                return_tensors=TensorType.NUMPY,
            )
            context["payload"] = payload
            context["pixel_values"] = inputs["pixel_values"]
            infer_inputs = []
            for key, input_tensor in inputs.items():
                if (not self.tensor_input_names) or (key in self.tensor_input_names):
                    infer_input = InferInput(
                        name=key,
                        datatype=from_np_dtype(input_tensor.dtype),
                        shape=list(input_tensor.shape),
                        data=input_tensor,
                    )
                    infer_inputs.append(infer_input)
            infer_request = InferRequest(
                infer_inputs=infer_inputs, model_name=self.name
            )
            return infer_request
        else:
            inputs = self._image_processor(
                instances[0],  # image
                return_tensors=TensorType.PYTORCH,
            )
            context["payload"] = payload
            context["pixel_values"] = inputs["pixel_values"]
            return inputs

    async def predict(
        self,
        input_batch: Union[Dict, InferRequest],
        context: Dict[str, Any],
    ) -> Union[Tensor, InferResponse]:
        if self.predictor_host:
            # when predictor_host is provided, serialize the tensor and send to optimized model serving runtime
            # like NVIDIA triton inference server
            return await super().predict(input_batch, context)
        else:  # local host
            input_batch = input_batch.to(self._device)
            try:
                with torch.no_grad():
                    if self._model is None:
                        raise ValueError()
                    outputs = self._model(**input_batch).logits
                    return outputs
            except Exception as e:
                raise InferenceError(str(e))

    def postprocess(
        self, outputs: Union[Tensor, InferResponse], context: Dict[str, Any]
    ) -> Union[Dict, InferResponse]:
        pixel_values = context["pixel_values"]
        request = context["payload"]
        if isinstance(outputs, InferResponse):
            shape = torch.Size(outputs.outputs[0].shape)
            data = torch.Tensor(outputs.outputs[0].data)
            outputs = data.view(shape)
            pixel_values = torch.Tensor(pixel_values)
        inferences = []
        if self.task == MLTask.image_classification:
            num_rows, num_cols = outputs.shape
            for i in range(num_rows):
                out = outputs[i].unsqueeze(0)
                predicted_idx = out.argmax().item()
                label = self.model_config.id2label[predicted_idx]
                # inferences.append(predicted_idx)
                inferences.append(label)
            return get_predict_response(request, inferences, self.name)
        else:
            raise ValueError(
                f"Unsupported task {self.task}. Please check the supported `task` option."
            )
