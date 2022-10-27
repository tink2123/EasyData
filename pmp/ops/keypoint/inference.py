# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from functools import reduce
import os
import importlib
import numpy as np
import math
import paddle
from pmp.ops.base import BaseOp
from pmp.ops.base import create_operators
from pmp.core.workspace import register
from .preprocess import *
from .postprocess import *


@register
class KeypointOp(BaseOp):
    def __init__(self, env_cfg, model_cfg):
        super().__init__(env_cfg, model_cfg)
        self.model_cfg = model_cfg
        mod = importlib.import_module(__name__)
        self.preprocessor = create_operators(model_cfg["PreProcess"], mod)
        self.postprocessor = create_operators(model_cfg["PostProcess"], mod)

    @classmethod
    def get_output_keys(cls):
        return ["keypoints", "kpt_scores"]

    def preprocess(self, image):
        im_info = {
            'im_shape': np.array(
                image.shape[:2], dtype=np.float32),
            'input_shape': self.model_cfg["image_shape"],
        }
        for ops in self.preprocessor:
            image, im_info = ops(image, im_info)
        return image, im_info

    def postprocess(self, inputs, im_shape, result):
        np_heatmap = result[0]
        im_shape = im_shape[:, ::-1]
        center = np.round(im_shape / 2.)
        scale = im_shape / 200.
        outputs = self.postprocessor[0](np_heatmap, center, scale,
                                        self.output_keys)
        return outputs

    def create_inputs(self, imgs, im_info):
        inputs = {}
        inputs = np.stack(imgs, axis=0).astype('float32')
        im_shape = []
        for e in im_info:
            im_shape.append(np.array((e['im_shape'])).astype('float32'))
        im_shape = np.stack(im_shape, axis=0)
        return inputs, im_shape

    def infer(self, image_list):
        inputs = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            output_list = []
            info_list = []
            for img in batch_image_list:
                output, info = self.preprocess(img)
                output_list.append(output)
                info_list.append(info)
            inputs, im_shape = self.create_inputs(output_list, info_list)

            # model inference
            result = self.predictor.run(inputs)

            # postprocess
            res = self.postprocess(inputs, im_shape, result)
            results.append(res)
        # results = self.merge_batch_result(results)
        return results

    def __call__(self, inputs):
        """
        step1: parser inputs
        step2: run
        step3: merge results
        input: a list of dict
        """
        # for the input_keys as list
        # inputs = [pipe_input[key] for pipe_input in pipe_inputs for key in self.input_keys]

        # step1: for the input_keys as str
        sub_index_list = [len(input)
                          for input in inputs]  # number of each batch
        inputs = reduce(lambda x, y: x.extend(y) or x, inputs)

        # step2: run
        outputs = self.infer(inputs)

        # step3: merge
        curr_offsef_id = 0
        pipe_outputs = []
        for idx in range(len(sub_index_list)):
            sub_start_idx = curr_offsef_id
            sub_end_idx = curr_offsef_id + sub_index_list[idx]
            output = outputs[sub_start_idx:sub_end_idx]
            output = {k: [o[k] for o in output] for k in output[0]}

            pipe_outputs.append(output)

            curr_offsef_id = sub_end_idx
        return pipe_outputs
