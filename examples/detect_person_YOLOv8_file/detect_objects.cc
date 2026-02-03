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

#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/gpio.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/camera/camera.h"
#include "libs/rpc/rpc_http_server.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/mjson/src/mjson.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/best_integer_quant_edgetpu.tflite";
constexpr char kImagePath[] =
    "/examples/detect_person_YOLOv8_file/frame_0003_320x320.rgb";
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 16 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

void DetectConsole(tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* in = interpreter->input(0);
  printf("in type=%d scale=%f zero=%ld dims=",
       (int)in->type, (double)in->params.scale, (long)in->params.zero_point);

  for (int i = 0; i < in->dims->size; ++i) printf("%d ", in->dims->data[i]);
  printf("\n");
  auto* input_tensor = interpreter->input_tensor(0);
  printf("Image exists? %d\r\n", coralmicro::LfsFileExists(kImagePath));
  printf("Image size: %d\r\n", (int)coralmicro::LfsSize(kImagePath));
  if (!LfsReadFile(kImagePath, tflite::GetTensorData<uint8_t>(input_tensor),
                   input_tensor->bytes)) {
    printf("ERROR: Failed to load %s\r\n", kImagePath);
    return;
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("ERROR: Invoke() failed\r\n");
    return;
  }
  auto results = tensorflow::GetDetectionResults(interpreter, 0.25f, 300);
  // auto results = tensorflow::GetDetectionResults(interpreter, 0.5, 1);
}

[[noreturn]] void Main() {
  printf("Run YOLOv8 with local file Example!\r\n");
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
  tflite::MicroMutableOpResolver<24> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddTranspose(); 
  resolver.AddSoftmax(); 
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddPad();
  resolver.AddResizeBilinear();
  resolver.AddQuantize();
  resolver.AddLogistic();
  resolver.AddCast();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddSub();
  resolver.AddStridedSlice(); 
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
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

  while (true) {
    auto start = TimerMicros();
    DetectConsole(&interpreter);
    auto end = TimerMicros();
    printf("total loop time: %d us\r\n", static_cast<int>((end - start)));
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
