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
    "/models/best_full_integer_quant_edgetpu.tflite";
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 16 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

bool DetectFromCamera(tflite::MicroInterpreter* interpreter, int model_width,
                      int model_height,
                      std::vector<tensorflow::Object>* results,
                      std::vector<uint8>* image) {
  CHECK(results != nullptr);
  CHECK(image != nullptr);
  auto* input_tensor = interpreter->input_tensor(0);
  CameraFrameFormat fmt{CameraFormat::kRgb,   CameraFilterMethod::kBilinear,
                        CameraRotation::k270, model_width,
                        model_height,         false,
                        image->data()};

  // Trigger camera and get frame.
  CameraTask::GetSingleton()->Trigger();
  if (!CameraTask::GetSingleton()->GetFrame({fmt})) return false;

  // copy image data to input tensor
  std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image->data(),
              image->size());
  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) return false;

  *results = tensorflow::GetDetectionResults(interpreter, 0.5, 1);
  return true;
}

void DetectConsole(tflite::MicroInterpreter* interpreter) {
  auto* input_tensor = interpreter->input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];
  std::vector<uint8> image(model_height * model_width *
                           CameraFormatBpp(CameraFormat::kRgb));
  std::vector<tensorflow::Object> results;
  if (DetectFromCamera(interpreter, model_width, model_height, &results,
                       &image)) {
    printf("%s\r\n", tensorflow::FormatDetectionOutput(results).c_str());
  } else {
    printf("Failed to detect image from camera.\r\n");
  }
}

[[noreturn]] void Main() {
  printf("Detection Camera Example!\r\n");
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

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kTrigger);

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
