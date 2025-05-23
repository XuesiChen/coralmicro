/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/person_detection/main_functions.h"

#include "libs/base/filesystem.h"
#include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"
#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "libs/base/timer.h" 

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

std::vector<uint8_t> g_person_detect_model_data_fs;
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 96 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  coralmicro::TimerInit();
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  if (!coralmicro::LfsReadFile("/models/person_detect_model.tflite",
                               &g_person_detect_model_data_fs)) {
    printf("Failed to read model\r\n");
    return;
  }

  model = tflite::GetModel(g_person_detect_model_data_fs.data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf(
        "Model provided is schema version %lu not equal "
        "to supported version %d.\r\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printf("AllocateTensors() failed\r\n");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  uint64_t preprocessing_start_time = coralmicro::TimerMicros();
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    printf("Image capture failed.\r\n");
  }
  uint64_t preprocessing_end_time = coralmicro::TimerMicros();
  uint64_t preprocessing_time = preprocessing_end_time - preprocessing_start_time;
  printf("Preprocessing time: %d us\r\n", static_cast<int>(preprocessing_time));

  // Run the model on this input and make sure it succeeds.
  uint64_t inference_start_time = coralmicro::TimerMicros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint64_t inference_end_time = coralmicro::TimerMicros();

  uint64_t inference_time = inference_end_time - inference_start_time;
  printf("Inference time: %d us\r\n", static_cast<int>(inference_time));

  uint64_t postprocessing_start_time = coralmicro::TimerMicros();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(error_reporter, person_score, no_person_score);

  uint64_t postprocessing_end_time = coralmicro::TimerMicros();
  uint64_t postprocessing_time = postprocessing_end_time - postprocessing_start_time;
  printf("Postprocessing time: %d us\r\n",
         static_cast<int>(postprocessing_time));
}
