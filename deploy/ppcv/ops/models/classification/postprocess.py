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

import os
import numpy as np


class Topk(object):
    def __init__(self, topk=1, class_id_map_file=None):
        assert isinstance(topk, (int, ))
        self.class_id_map = self.parse_class_id_map(class_id_map_file)
        self.topk = topk

    def parse_class_id_map(self, class_id_map_file):
        if class_id_map_file is None:
            return None

        if not os.path.exists(class_id_map_file):
            print(
                "Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
            )
            return None

        try:
            class_id_map = {}
            with open(class_id_map_file, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    partition = line.split("\n")[0].partition(" ")
                    class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            print(ex)
            class_id_map = None
        return class_id_map

    def __call__(self, x, output_keys):
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                output_keys[0]: clas_id_list,
                output_keys[1]: np.around(
                    score_list, decimals=5).tolist(),
            }
            if label_name_list is not None:
                result[output_keys[2]] = label_name_list
            y.append(result)
        return y
