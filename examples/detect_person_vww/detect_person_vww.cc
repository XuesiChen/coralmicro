// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include <math.h>
#include "libs/camera/camera.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/vww_96_int8_edgetpu.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  coralmicro::TimerInit();
  
  printf("Binary Person Detection Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddDequantize();
  resolver.AddCustom(kCustomOp, RegisterCustomOp()); 

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  auto* input_tensor = interpreter.input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];

  printf("Input tensor type: %d\r\n", input_tensor->type);
  printf("Model expects: %d x %d x %d\r\n",
       input_tensor->dims->data[1],
       input_tensor->dims->data[2],
       input_tensor->dims->data[3]);

  while (true) {
    // -- PRE-PROCESSING : --
    auto preprocessing_start_us = coralmicro::TimerMicros();
    // --- code dani added : end ---
    CameraFrameFormat fmt{CameraFormat::kRgb,
                          CameraFilterMethod::kBilinear,
                          CameraRotation::k270,
                          model_width,
                          model_height,
                          false,
                          tflite::GetTensorData<uint8_t>(input_tensor)};
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("Failed to capture image\r\n");
      vTaskSuspend(nullptr);
    }
    auto preprocessing_end_us = coralmicro::TimerMicros();
    // -- INFERENCE : --
    auto inference_start_us = coralmicro::TimerMicros();
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }
    auto inference_end_us = coralmicro::TimerMicros();
    // -- POST-PROCESSING : --
    auto postprocessing_start_us = coralmicro::TimerMicros();
    // Visual Wake Words is a binary classifier:
    // Output tensor is [score for "no person", score for "person"]
    auto* output = interpreter.output_tensor(0);
    int8_t raw_no_person = output->data.int8[0];
    int8_t raw_person = output->data.int8[1];
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    float prob_no_person = (raw_no_person - zero_point) * scale;
    float prob_person = (raw_person - zero_point) * scale;
    auto postprocessing_end_us = coralmicro::TimerMicros();
    if (prob_person > 0.5f) {
      printf("PERSON DETECTED!\r\n");
      LedSet(Led::kUser, true);
    } else {
      LedSet(Led::kUser, false);
    }
    printf("Timing (us):   Preprocessing=%lu\t, Inference=%lu\t, Postprocessing=%lu\r\n",
      static_cast<uint32_t>(preprocessing_end_us - preprocessing_start_us), static_cast<uint32_t>(inference_end_us - inference_start_us), static_cast<uint32_t>(postprocessing_end_us - postprocessing_start_us));
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
