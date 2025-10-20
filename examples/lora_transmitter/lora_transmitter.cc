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
/*
*/
#include <vector>
#include "libs/base/filesystem.h"
#include "libs/base/led.h"
// --- code dani added : start ---
#include "libs/base/timer.h"
// --- code dani added : end ---
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

// Runs face detection on the Edge TPU, using the on-board camera, printing
// results to the serial console and turning on the User LED when a face
// is detected.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e face_detection

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  // --- code dani added : start ---
  coralmicro::TimerInit();
  // --- code dani added : end ---
  
  //printf("LoRa Transmitter Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    //printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    //printf("ERROR: Fprintf("Found %d face(s):\r\n%s\r\n", results.size(),
    //         tensorflow::FormatDetectionOutput(results).c_str());iled to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    //printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    //printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  auto* input_tensor = interpreter.input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];

  // --- code dani added : start ---
  constexpr int kReceiverAddress = 473;
  // --- code dani added : end ---

  while (true) {
    // --- code dani added : start ---
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
      //printf("Failed to capture image\r\n");
      vTaskSuspend(nullptr);
    }
    // --- code dani added : start ---
    auto preprocessing_end_us = coralmicro::TimerMicros();
    // -- INFERENCE : --
    auto inference_start_us = coralmicro::TimerMicros();
    // --- code dani added : end ---
    if (interpreter.Invoke() != kTfLiteOk) {
      //printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }
    // --- code dani added : start ---
    auto inference_end_us = coralmicro::TimerMicros();
    // -- POST-PROCESSING : --
    auto postprocessing_start_us = coralmicro::TimerMicros();
    auto results = tensorflow::GetDetectionResults(&interpreter, kThreshold, kTopK);
    auto postprocessing_end_us = coralmicro::TimerMicros();
    // --- code dani added : end ---
    if (!results.empty()) { // I (Dani) changed this line in accordance to the line I added above.
      //printf("Found %d face(s):\r\n%s\r\n", results.size(),
      //       tensorflow::FormatDetectionOutput(results).c_str());
      LedSet(Led::kUser, true);
    } else {
      LedSet(Led::kUser, false);
      //printf("Found %d face(s):\r\n%s\r\n", results.size(),
      //       tensorflow::FormatDetectionOutput(results).c_str());
    }
    // --- code dani added : start ---
    // send the number of detected faces via lora transmition:
    int face_count = static_cast<int>(results.size());
    char payload[16];
    snprintf(payload, sizeof(payload), "%d", face_count);
    int data_length = strlen(payload);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "AT+SEND=%d,%d,%s\r\n", kReceiverAddress, data_length, payload);
    // Send command over UART6 (console)
    printf("%s", cmd); 
    // --- code dani added : end ---
    // --- code dani added : start ---
    //printf("Timing (us):   Preprocessing=%lu\t, Inference=%lu\t, Postprocessing=%lu\r\n",
    //  static_cast<uint32_t>(preprocessing_end_us - preprocessing_start_us), static_cast<uint32_t>(inference_end_us - inference_start_us), static_cast<uint32_t>(postprocessing_end_us - postprocessing_start_us));
    /*
    printf("|----------------------------------------------------------------------------------\r\n");
    printf("|               | Preprocessing       | Inference           | Postprocessing      |\r\n");
    printf("|---------------|---------------------|---------------------|---------------------|\r\n");
    printf("| Start Time:   | %-20lu| %-20lu| %-20lu|\r\n",static_cast<uint32_t>(preprocessing_start_us),static_cast<uint32_t>(inference_start_us),static_cast<uint32_t>(postprocessing_start_us));
    printf("|---------------|---------------------|---------------------|---------------------|\r\n");
    printf("| End Time:     | %-20lu| %-20lu| %-20lu|\r\n",static_cast<uint32_t>(preprocessing_end_us),static_cast<uint32_t>(inference_end_us),static_cast<uint32_t>(postprocessing_end_us));
    printf("-----------------------------------------------------------------------------------\r\n");
    */
    // --- code dani added : end ---
    // --- code dani added : start ---
    vTaskDelay(pdMS_TO_TICKS(1000)); // wait 1000 ms
    // --- code dani added : end ---
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
