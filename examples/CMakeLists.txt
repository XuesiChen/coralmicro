# Copyright 2022 Google LLC
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

add_subdirectory(classify_speech_MLPerfTiny_kws)
add_subdirectory(classify_speech_arduino_kws)
add_subdirectory(detect_person_vww)
add_subdirectory(detect_person_YOLOv8_with_TPU)
add_subdirectory(detect_person_YOLOv8_without_TPU)
add_subdirectory(detect_person_MobileNetV2_with_TPU)
add_subdirectory(detect_person_MobileNetV2_without_TPU)
add_subdirectory(tflm_person_detection_m7)
