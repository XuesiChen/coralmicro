# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
  $(error "Requires make version 3.82 or later (current is $(MAKE_VERSION))")
endif

# root directory of tensorflow
TENSORFLOW_ROOT :=
MAKEFILE_DIR := tensorflow/lite/micro/tools/make

# Pull in some convenience functions.
include $(MAKEFILE_DIR)/helper_functions.inc

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

# Determine the host architecture, with any ix86 architecture being labelled x86_32
HOST_ARCH := $(shell if uname -m | grep -Eq 'i[345678]86'; then echo x86_32; else echo $(shell uname -m); fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

# Default compiler and tool names:
TOOLCHAIN:=gcc
CXX_TOOL := g++
CC_TOOL := gcc
AR_TOOL := ar

ifneq ($(TAGS),)
  $(error The TAGS command line option is no longer supported in the TFLM Makefile.)
endif

# Specify which specialized kernel implementation should be pulled in.
OPTIMIZED_KERNEL_DIR :=

# Override this variable from the command line in case the optimized kernels are
# in a different directory.
OPTIMIZED_KERNEL_DIR_PREFIX := tensorflow/lite/micro/kernels

# Specify which co-processor's kernel implementation should be pulled in.
# If the same kernel is implemented in both kernels/OPTIMIZED_KERNEL_DIR and
# kernels/CO_PROCESSOR, then the implementation from kernels/CO_PROCESSOR will
# be used.
CO_PROCESSOR :=

# This is obviously horrible.  We need to generate these 3 versions of the
# include directories from one source.
INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(MAKEFILE_DIR)/downloads/ruy

# Same list of paths, but now relative to the generated project files.
GENERATED_PROJECT_INCLUDES := \
-I. \
-I./third_party/gemmlowp \
-I./third_party/flatbuffers/include \
-I./third_party/ruy

# Same list of paths, but now in the format the generate_keil_project.py
# script expects them.
PROJECT_INCLUDES := \
. \
third_party/gemmlowp \
third_party/flatbuffers/include \
third_party/ruy

TEST_SCRIPT :=

MICROLITE_LIBS := -lm

# For the optimized_kernel_dir, and co-processor as specified on the
# command line we add -D<tag> to the cflags to allow for #idefs in the code.
#
# We apply the following transformations (via the tr command):
#   1. Convert to uppercase (OPTIMIZED_KERNEL_DIR=xtensa -> -DXTENSA)
ADDITIONAL_DEFINES :=
ifneq ($(OPTIMIZED_KERNEL_DIR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZED_KERNEL_DIR) | tr [a-z] [A-Z])
endif

ifneq ($(CO_PROCESSOR),)
  ADDITIONAL_DEFINES += -D$(shell echo $(CO_PROCESSOR) | tr [a-z] [A-Z])
endif

ifeq ($(TOOLCHAIN), armclang)
  CORE_OPTIMIZATION_LEVEL := -Oz
else
  CORE_OPTIMIZATION_LEVEL := -Os
endif
KERNEL_OPTIMIZATION_LEVEL := -O2
THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL := -O2

# Warn if deprecated optimization level is set.
OPTIMIZATION_LEVEL :=
ifneq ($(OPTIMIZATION_LEVEL),)
$(error "OPTIMIZATION_LEVEL is no longer used.")
endif


CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wshadow \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

COMMON_FLAGS := \
  -Werror \
  -fno-unwind-tables \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  $(CC_WARNINGS) \
  $(ADDITIONAL_DEFINES)

ifeq ($(TARGET), $(HOST_OS))
  # If we are not doing a cross-compilation then -DTF_LITE_USE_CTIME is what we
  # want to have by default.
  COMMON_FLAGS += -DTF_LITE_USE_CTIME
endif

CXXFLAGS := \
  -std=c++11 \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  $(COMMON_FLAGS)

CCFLAGS := \
  -Wimplicit-function-declaration \
  -std=c11 \
  $(COMMON_FLAGS)

ARFLAGS := -r

ifeq ($(TOOLCHAIN), gcc)
  ifneq ($(TARGET), osx)
    # GCC on MacOS uses an LLVM backend so we avoid the additional linker flags
    # that are unsupported with LLVM.
    LDFLAGS += \
      -Wl,--fatal-warnings \
      -Wl,--gc-sections
  endif
endif

# override these in the makefile.inc for specific compiler targets
TARGET_TOOLCHAIN_PREFIX :=
TARGET_TOOLCHAIN_ROOT :=

# Specifying BUILD_TYPE=<blah> as part of the make command gives us a few
# options to choose from.
#
# If BUILD_TYPE is not specified, the default build (which should be suitable
# most of the time) has all of the error checking logic at the expense of a
# latency increase of ~5-10% relative to BUILD_TYPE=release_with_logs.
#
# This default build is most suited for usual development and testing as is
# highlighted by the discussion on this github pull request:
# https://github.com/tensorflow/tensorflow/pull/42314#issuecomment-694360567
BUILD_TYPE := default
ifeq ($(BUILD_TYPE), debug)
	# Specifying BUILD_TYPE=debug adds debug symbols to the binary (and makes it
	# larger) and should be used to run a binary with gdb.
	CXXFLAGS += -g
	CCFLAGS  += -g
else ifeq ($(BUILD_TYPE), release)
	# The 'release' build results in the smallest binary (by virtue of removing
	# strings from log messages, DCHECKs ...).
	#
	# The down-side is that we currently do not have a good mechanism to allow
	# for logging that is not related to errors (e.g. profiling information, or
	# logs that help determine if tests pass or fail). As a result, we are unable
	# to run tests or benchmarks with BUILD_TYPE=release (which is a bit
	# counter-intuitive). TODO(b/158205789): A global error reporter might help.
	#
	# For a close approximation of the release build use
	# BUILD_TYPE=release_with_logs.
	CXXFLAGS += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
	CCFLAGS  += -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
else ifeq ($(BUILD_TYPE), release_with_logs)
	# The latency with BUILD_TYPE=release_with_logs will be close to the 'release'
	# build and there will still be error logs. This build type may be preferable
	# for profiling and benchmarking.
	CXXFLAGS += -DNDEBUG
	CCFLAGS  += -DNDEBUG
else ifeq ($(BUILD_TYPE), no_tf_lite_static_memory)
	# This build should not be used to run any binaries/tests since
	# TF_LITE_STATIC_MEMORY should be defined for all micro builds. However,
	# having a build without TF_LITE_STATIC_MEMORY is useful to catch errors in
	# code that is shared between TfLite Mobile and TfLite Micro. See this issue
	# for more details:
	# https://github.com/tensorflow/tensorflow/issues/43076
	CXXFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CXXFLAGS))
	CCFLAGS := $(filter-out -DTF_LITE_STATIC_MEMORY, $(CCFLAGS))
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
MICROLITE_LIB_NAME := libtensorflow-microlite.a

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET)_$(TARGET_ARCH)_$(BUILD_TYPE)/
CORE_OBJDIR := $(GENDIR)obj/core/
KERNEL_OBJDIR := $(GENDIR)obj/kernels/
THIRD_PARTY_KERNEL_OBJDIR := $(GENDIR)obj/third_party_kernels/
THIRD_PARTY_OBJDIR := $(GENDIR)obj/third_party/
GENERATED_SRCS_DIR := $(GENDIR)genfiles/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/
PRJDIR := $(GENDIR)prj/

# These two must be defined before we include the target specific Makefile.inc
# because we filter out the examples that are not supported for those targets.
# See targets/xtensa_xpg_makefile.inc for an example.
#
# We limit max depth of directories to search to not include target specific
# Makefiles that are included directly by the main example Makefile. See
# examples/micro_speech/Makefile.inc for an example. At the same time, we
# search till an arbitrary depth for files named Makefile_internal.inc as a way
# to bypass this check and allow for deeper directory structures.
MICRO_LITE_EXAMPLE_TESTS := $(shell find tensorflow/lite/micro/examples/ -maxdepth 2 -name Makefile.inc)
# Internal examples are copied outside the TFLM repo in ../google.
ifneq ($(wildcard ../google),)
MICRO_LITE_EXAMPLE_TESTS += $(shell find ../google/ -name Makefile_internal.inc)
endif

# Kernel integration tests must be excluded on certain targets.
MICRO_LITE_INTEGRATION_TESTS += $(shell find tensorflow/lite/micro/integration_tests -name Makefile.inc)

MICRO_LITE_BENCHMARKS := $(wildcard tensorflow/lite/micro/benchmarks/Makefile.inc)

# TODO(b/152645559): move all benchmarks to benchmarks directory.
MICROLITE_BENCHMARK_SRCS := \
$(wildcard tensorflow/lite/micro/benchmarks/*benchmark.cc)

MICROLITE_TEST_SRCS := \
tensorflow/lite/micro/fake_micro_context_test.cc \
tensorflow/lite/micro/flatbuffer_utils_test.cc \
tensorflow/lite/micro/memory_arena_threshold_test.cc \
tensorflow/lite/micro/memory_helpers_test.cc \
tensorflow/lite/micro/micro_allocator_test.cc \
tensorflow/lite/micro/micro_allocation_info_test.cc \
tensorflow/lite/micro/micro_context_test.cc \
tensorflow/lite/micro/micro_error_reporter_test.cc \
tensorflow/lite/micro/micro_interpreter_test.cc \
tensorflow/lite/micro/micro_mutable_op_resolver_test.cc \
tensorflow/lite/micro/micro_resource_variable_test.cc \
tensorflow/lite/micro/micro_string_test.cc \
tensorflow/lite/micro/micro_time_test.cc \
tensorflow/lite/micro/micro_utils_test.cc \
tensorflow/lite/micro/recording_micro_allocator_test.cc \
tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator_test.cc \
tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator_test.cc \
tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator_test.cc \
tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator_test.cc \
tensorflow/lite/micro/testing_helpers_test.cc \
tensorflow/lite/micro/memory_planner/greedy_memory_planner_test.cc \
tensorflow/lite/micro/memory_planner/linear_memory_planner_test.cc \
tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim_test.cc

MICROLITE_CC_KERNEL_SRCS := \
tensorflow/lite/micro/kernels/activations.cc \
tensorflow/lite/micro/kernels/activations_common.cc \
tensorflow/lite/micro/kernels/add.cc \
tensorflow/lite/micro/kernels/add_common.cc \
tensorflow/lite/micro/kernels/add_n.cc \
tensorflow/lite/micro/kernels/arg_min_max.cc \
tensorflow/lite/micro/kernels/assign_variable.cc \
tensorflow/lite/micro/kernels/batch_to_space_nd.cc \
tensorflow/lite/micro/kernels/broadcast_args.cc \
tensorflow/lite/micro/kernels/broadcast_to.cc \
tensorflow/lite/micro/kernels/call_once.cc \
tensorflow/lite/micro/kernels/cast.cc \
tensorflow/lite/micro/kernels/ceil.cc \
tensorflow/lite/micro/kernels/circular_buffer.cc \
tensorflow/lite/micro/kernels/circular_buffer_common.cc \
tensorflow/lite/micro/kernels/comparisons.cc \
tensorflow/lite/micro/kernels/concatenation.cc \
tensorflow/lite/micro/kernels/conv.cc \
tensorflow/lite/micro/kernels/conv_common.cc \
tensorflow/lite/micro/kernels/cumsum.cc \
tensorflow/lite/micro/kernels/depth_to_space.cc \
tensorflow/lite/micro/kernels/depthwise_conv.cc \
tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
tensorflow/lite/micro/kernels/dequantize.cc \
tensorflow/lite/micro/kernels/dequantize_common.cc \
tensorflow/lite/micro/kernels/detection_postprocess.cc \
tensorflow/lite/micro/kernels/div.cc \
tensorflow/lite/micro/kernels/elementwise.cc \
tensorflow/lite/micro/kernels/elu.cc \
tensorflow/lite/micro/kernels/ethosu.cc \
tensorflow/lite/micro/kernels/exp.cc \
tensorflow/lite/micro/kernels/expand_dims.cc \
tensorflow/lite/micro/kernels/fill.cc \
tensorflow/lite/micro/kernels/floor.cc \
tensorflow/lite/micro/kernels/floor_div.cc \
tensorflow/lite/micro/kernels/floor_mod.cc \
tensorflow/lite/micro/kernels/fully_connected.cc \
tensorflow/lite/micro/kernels/fully_connected_common.cc \
tensorflow/lite/micro/kernels/gather.cc \
tensorflow/lite/micro/kernels/gather_nd.cc \
tensorflow/lite/micro/kernels/hard_swish.cc \
tensorflow/lite/micro/kernels/hard_swish_common.cc \
tensorflow/lite/micro/kernels/if.cc \
tensorflow/lite/micro/kernels/kernel_runner.cc \
tensorflow/lite/micro/kernels/kernel_util.cc \
tensorflow/lite/micro/kernels/l2norm.cc \
tensorflow/lite/micro/kernels/l2_pool_2d.cc \
tensorflow/lite/micro/kernels/leaky_relu.cc \
tensorflow/lite/micro/kernels/leaky_relu_common.cc \
tensorflow/lite/micro/kernels/logical.cc \
tensorflow/lite/micro/kernels/logical_common.cc \
tensorflow/lite/micro/kernels/logistic.cc \
tensorflow/lite/micro/kernels/logistic_common.cc \
tensorflow/lite/micro/kernels/log_softmax.cc \
tensorflow/lite/micro/kernels/lstm_eval.cc \
tensorflow/lite/micro/kernels/maximum_minimum.cc \
tensorflow/lite/micro/kernels/micro_tensor_utils.cc \
tensorflow/lite/micro/kernels/mirror_pad.cc \
tensorflow/lite/micro/kernels/mul.cc \
tensorflow/lite/micro/kernels/mul_common.cc \
tensorflow/lite/micro/kernels/neg.cc \
tensorflow/lite/micro/kernels/pack.cc \
tensorflow/lite/micro/kernels/pad.cc \
tensorflow/lite/micro/kernels/pooling.cc \
tensorflow/lite/micro/kernels/pooling_common.cc \
tensorflow/lite/micro/kernels/prelu.cc \
tensorflow/lite/micro/kernels/prelu_common.cc \
tensorflow/lite/micro/kernels/quantize.cc \
tensorflow/lite/micro/kernels/quantize_common.cc \
tensorflow/lite/micro/kernels/read_variable.cc \
tensorflow/lite/micro/kernels/reduce.cc \
tensorflow/lite/micro/kernels/reduce_common.cc \
tensorflow/lite/micro/kernels/reshape.cc \
tensorflow/lite/micro/kernels/resize_bilinear.cc \
tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
tensorflow/lite/micro/kernels/round.cc \
tensorflow/lite/micro/kernels/shape.cc \
tensorflow/lite/micro/kernels/slice.cc \
tensorflow/lite/micro/kernels/softmax.cc \
tensorflow/lite/micro/kernels/softmax_common.cc \
tensorflow/lite/micro/kernels/space_to_batch_nd.cc \
tensorflow/lite/micro/kernels/space_to_depth.cc \
tensorflow/lite/micro/kernels/split.cc \
tensorflow/lite/micro/kernels/split_v.cc \
tensorflow/lite/micro/kernels/squared_difference.cc \
tensorflow/lite/micro/kernels/squeeze.cc \
tensorflow/lite/micro/kernels/strided_slice.cc \
tensorflow/lite/micro/kernels/sub.cc \
tensorflow/lite/micro/kernels/sub_common.cc \
tensorflow/lite/micro/kernels/svdf.cc \
tensorflow/lite/micro/kernels/svdf_common.cc \
tensorflow/lite/micro/kernels/tanh.cc \
tensorflow/lite/micro/kernels/transpose.cc \
tensorflow/lite/micro/kernels/transpose_conv.cc \
tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cc \
tensorflow/lite/micro/kernels/unpack.cc \
tensorflow/lite/micro/kernels/var_handle.cc \
tensorflow/lite/micro/kernels/while.cc \
tensorflow/lite/micro/kernels/zeros_like.cc

MICROLITE_TEST_HDRS := \
$(wildcard tensorflow/lite/micro/testing/*.h)

# The explicitly specified list of sources and headers that are shared between
# TfLite and TFLM are in the ci/sync_from_upstream_tf.sh script.
TFL_CC_SRCS := \
$(shell find tensorflow/lite -type d \( -path tensorflow/lite/experimental -o -path tensorflow/lite/micro \) -prune -false -o -name "*.cc" -o -name "*.c")

TFL_CC_HDRS := \
$(shell find tensorflow/lite -type d \( -path tensorflow/lite/experimental -o -path tensorflow/lite/micro \) -prune -false -o -name "*.h")

MICROLITE_CC_BASE_SRCS := \
$(wildcard tensorflow/lite/micro/*.cc) \
$(wildcard tensorflow/lite/micro/arena_allocator/*.cc) \
$(wildcard tensorflow/lite/micro/memory_planner/*.cc) \
$(TFL_CC_SRCS)

MICROLITE_CC_HDRS := \
$(wildcard tensorflow/lite/micro/*.h) \
$(wildcard tensorflow/lite/micro/benchmarks/*model_data.h) \
$(wildcard tensorflow/lite/micro/kernels/*.h) \
$(wildcard tensorflow/lite/micro/arena_allocator/*.h) \
$(wildcard tensorflow/lite/micro/memory_planner/*.h) \
LICENSE \
$(TFL_CC_HDRS)

# TODO(b/165940489): Figure out how to avoid including fixed point
# platform-specific headers.
# some kiss fft source file has to be included in header sections because
# the implemenation of three different resolution fft from one single c file.
# See http://b/201319430 for additional context.
THIRD_PARTY_CC_HDRS := \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/allocator.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/array.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/base.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/buffer.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/buffer_ref.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/default_allocator.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/detached_buffer.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/flatbuffer_builder.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/flatbuffers.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/flexbuffers.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/stl_emulation.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/string.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/struct.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/table.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/vector.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/vector_downward.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/verifier.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/include/flatbuffers/util.h \
$(MAKEFILE_DIR)/downloads/flatbuffers/LICENSE.txt \
$(MAKEFILE_DIR)/downloads/gemmlowp/fixedpoint/fixedpoint.h \
$(MAKEFILE_DIR)/downloads/gemmlowp/fixedpoint/fixedpoint_neon.h \
$(MAKEFILE_DIR)/downloads/gemmlowp/fixedpoint/fixedpoint_sse.h \
$(MAKEFILE_DIR)/downloads/gemmlowp/internal/detect_platform.h \
$(MAKEFILE_DIR)/downloads/gemmlowp/LICENSE \
$(MAKEFILE_DIR)/downloads/kissfft/COPYING \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.c \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.h \
$(MAKEFILE_DIR)/downloads/kissfft/_kiss_fft_guts.h \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.c \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.h \
$(MAKEFILE_DIR)/downloads/ruy/ruy/profiler/instrumentation.h

THIRD_PARTY_CC_SRCS :=
THIRD_PARTY_KERNEL_CC_SRCS :=

# Load custom kernels.
include $(MAKEFILE_DIR)/additional_kernels.inc

MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_TEST_SRCS), $(MICROLITE_CC_BASE_SRCS))
MICROLITE_CC_SRCS := $(filter-out $(MICROLITE_BENCHMARK_SRCS), $(MICROLITE_CC_SRCS))



# The download scripts require that the downloads directory already exist for
# improved error checking. To accomodate that, we first create a downloads
# directory.
$(shell mkdir -p ${MAKEFILE_DIR}/downloads)

# Directly download the flatbuffers library.
DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/flatbuffers_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the flatbuffers download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/kissfft_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the kissfft download: $(DOWNLOAD_RESULT))
endif

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/pigweed_download.sh ${MAKEFILE_DIR}/downloads)
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the pigweed download: $(DOWNLOAD_RESULT))
endif

include $(MAKEFILE_DIR)/third_party_downloads.inc
THIRD_PARTY_DOWNLOADS :=
$(eval $(call add_third_party_download,$(GEMMLOWP_URL),$(GEMMLOWP_MD5),gemmlowp,))
$(eval $(call add_third_party_download,$(RUY_URL),$(RUY_MD5),ruy,))

# The target-specific makefile must have a name that is exactly
# TARGET_makefile.inc and is only needed for cross-compilation (i.e. when TARGET
# is different from the HOST_OS).
TARGETS_WITHOUT_MAKEFILES := \
$(HOST_OS)

# This specific string needs to be outputted for a test to be recognized as
# having passed.
TEST_PASS_STRING:='~~~ALL TESTS PASSED~~~'

# ${TARGET}_makefile.inc can set this to true to allow it to defined a custom
# implementation for `make test`. See bluepill_makefile as an example.
TARGET_SPECIFIC_MAKE_TEST:=0

ifeq ($(findstring $(TARGET),$(TARGETS_WITHOUT_MAKEFILES)),)
  include $(MAKEFILE_DIR)/targets/$(TARGET)_makefile.inc
endif

ifneq ($(OPTIMIZED_KERNEL_DIR),)
  PATH_TO_OPTIMIZED_KERNELS := $(OPTIMIZED_KERNEL_DIR_PREFIX)/$(OPTIMIZED_KERNEL_DIR)

  # Check that OPTIMIZED_KERNEL_DIR is valid to avoid unexpected fallback to
  # reference kernels. See http://b/183546742 for more context.
  RESULT := $(shell $(MAKEFILE_DIR)/check_optimized_kernel_dir.sh $(PATH_TO_OPTIMIZED_KERNELS))
  ifneq ($(RESULT), SUCCESS)
    $(error Incorrect OPTIMIZED_KERNEL_DIR: $(RESULT))
  endif

  include $(MAKEFILE_DIR)/ext_libs/$(OPTIMIZED_KERNEL_DIR).inc
  # Specialize for the optimized kernels
  MICROLITE_CC_KERNEL_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
		--base_files "$(MICROLITE_CC_KERNEL_SRCS)" \
		--specialize_directory $(PATH_TO_OPTIMIZED_KERNELS))

  # The first ifneq is needed to be compatible with make versions prior to 4.2
  # which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
  # Ubuntu 18.04 only has version 4.1
  ifneq ($(.SHELLSTATUS),)
    ifneq ($(.SHELLSTATUS),0)
      $(error Error with specialize_files.py $(MICROLITE_CC_KERNEL_SRCS))
    endif
  endif

  # Optimized kernel directories can have their own header files which need to
  # be included in MICROLITE_CC_HDRS for project generation to have a complete
  # list of headers.
  MICROLITE_CC_HDRS += $(wildcard $(PATH_TO_OPTIMIZED_KERNELS)/*.h)
endif

# If a co-processor is specified on the command line with
# CO_PROCESSOR=<co_processor> then we will include ext_libs/<co_processor>.inc
# and find additional kernel sources in kernels/<co_processor>/
#
# That the co-processor specialization of the kernel sources happens after the
# optimized_kernel_dir means that if there is an implementation of the same
# kernel in both directories, the one from co_processor will be used.
ifneq ($(CO_PROCESSOR),)
  include $(MAKEFILE_DIR)/ext_libs/$(CO_PROCESSOR).inc
  # Specialize for the coprocessor kernels.
  PATH_TO_COPROCESSOR_KERNELS := tensorflow/lite/micro/kernels/$(CO_PROCESSOR)
  MICROLITE_CC_KERNEL_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
    --base_files "$(MICROLITE_CC_KERNEL_SRCS)" \
    --specialize_directory $(PATH_TO_COPROCESSOR_KERNELS))

  # The first ifneq is needed to be compatible with make versions prior to 4.2
  # which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
  # Ubuntu 18.04 only has version 4.1
  ifneq ($(.SHELLSTATUS),)
    ifneq ($(.SHELLSTATUS),0)
      $(error Error with specialize_files.py $(MICROLITE_CC_KERNEL_SRCS))
    endif
  endif
endif

# Specialize for debug_log. micro_time etc.
PATH_TO_TARGET_SRCS := tensorflow/lite/micro/$(TARGET)
MICROLITE_CC_SRCS := $(shell python3 $(MAKEFILE_DIR)/specialize_files.py \
  --base_files "$(MICROLITE_CC_SRCS)" \
  --specialize_directory $(PATH_TO_TARGET_SRCS))

# The first ifneq is needed to be compatible with make versions prior to 4.2
# which do not support .SHELLSTATUS. While make 4.2 was released in 2016,
# Ubuntu 18.04 only has version 4.1
ifneq ($(.SHELLSTATUS),)
  ifneq ($(.SHELLSTATUS),0)
    $(error Error with specialize_files.py $(MICROLITE_CC_SRCS))
  endif
endif

ALL_SRCS := \
	$(MICROLITE_CC_SRCS) \
	$(MICROLITE_CC_KERNEL_SRCS) \
	$(MICROLITE_TEST_SRCS)

MICROLITE_LIB_PATH := $(LIBDIR)$(MICROLITE_LIB_NAME)

CXX := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CXX_TOOL}
CC := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${CC_TOOL}
AR := $(TARGET_TOOLCHAIN_ROOT)${TARGET_TOOLCHAIN_PREFIX}${AR_TOOL}

# The default Makefile target(all) must appear before any target,
# which is compiled if there's no command-line arguments.
all: $(MICROLITE_LIB_PATH)

# Include output directory since example cc files depend on generated headers.
INCLUDES += -I$(GENERATED_SRCS_DIR)

# Load the examples.
include $(MICRO_LITE_EXAMPLE_TESTS)

# Load the integration tests.
include $(MICRO_LITE_INTEGRATION_TESTS)

# Load the benchmarks.
include $(MICRO_LITE_BENCHMARKS)

# Load custom kernel tests.
include $(MAKEFILE_DIR)/additional_tests.inc

# Create rules for downloading third-party dependencies.
THIRD_PARTY_TARGETS :=
$(foreach DOWNLOAD,$(THIRD_PARTY_DOWNLOADS),$(eval $(call create_download_rule,$(DOWNLOAD))))
third_party_downloads: $(THIRD_PARTY_TARGETS)

MICROLITE_LIB_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_SRCS)))))

MICROLITE_THIRD_PARTY_OBJS := $(addprefix $(THIRD_PARTY_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(THIRD_PARTY_CC_SRCS)))))

MICROLITE_THIRD_PARTY_KERNEL_OBJS := $(addprefix $(THIRD_PARTY_KERNEL_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(THIRD_PARTY_KERNEL_CC_SRCS)))))

MICROLITE_KERNEL_OBJS := $(addprefix $(KERNEL_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICROLITE_CC_KERNEL_SRCS)))))

$(CORE_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(CORE_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(CORE_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(CORE_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_KERNEL_OBJDIR)%.o: %.cc $(THIRD_PARTY_KERNEL_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_KERNEL_OBJDIR)%.o: %.c $(THIRD_PARTY_KERNEL_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(THIRD_PARTY_KERNEL_OBJDIR)%.o: %.S $(THIRD_PARTY_KERNEL_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.cc $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.c $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

$(KERNEL_OBJDIR)%.o: %.S $(THIRD_PARTY_TARGETS)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(KERNEL_OPTIMIZATION_LEVEL) $(INCLUDES) -c $< -o $@

microlite: $(MICROLITE_LIB_PATH)

# Gathers together all the objects we've compiled into a single '.a' archive.
$(MICROLITE_LIB_PATH): $(MICROLITE_LIB_OBJS) $(MICROLITE_KERNEL_OBJS) $(MICROLITE_THIRD_PARTY_OBJS) $(MICROLITE_THIRD_PARTY_KERNEL_OBJS) $(MICROLITE_CUSTOM_OP_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MICROLITE_LIB_PATH) $(MICROLITE_LIB_OBJS) \
	$(MICROLITE_KERNEL_OBJS) $(MICROLITE_THIRD_PARTY_OBJS) $(MICROLITE_THIRD_PARTY_KERNEL_OBJS) $(MICROLITE_CUSTOM_OP_OBJS)

$(BINDIR)%_test : $(CORE_OBJDIR)%_test.o $(MICROLITE_LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $< \
	$(MICROLITE_LIB_PATH) $(LDFLAGS) $(MICROLITE_LIBS)

$(BINDIR)%.test_target: $(BINDIR)%_test
	@test -f $(TEST_SCRIPT) || (echo 'Unable to find the test script. Is the software emulation available in $(TARGET)?'; exit 1)
	$(TEST_SCRIPT) $< $(TEST_PASS_STRING)

# snease: Add %.bin rule here since BINDIR is now defined
# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
ifeq ($(TOOLCHAIN), armclang)
  FROMELF := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)fromelf
  $(BINDIR)%.bin: $(BINDIR)%
		@mkdir -p $(dir $@)
		$(FROMELF) --bin --output=$@ $<
else
  OBJCOPY := ${TARGET_TOOLCHAIN_ROOT}$(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)%.bin: $(BINDIR)%
		@mkdir -p $(dir $@)
		$(OBJCOPY) $< $@ -O binary
endif

# Create kernel test targets.
include tensorflow/lite/micro/kernels/Makefile.inc

# Create binary size test target.
include tensorflow/lite/micro/tools/ci_build/binary_size_test/Makefile.inc

# Some tests have additional dependencies (beyond libtensorflow-microlite.a) and
# those need to be explicitly specified with their own individual call to the
# microlite_test helper function. For these tests, we also need to make sure to
# not add targets for them if they have been excluded as part of the target
# specific Makefile.
EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/memory_arena_threshold_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/testing/test_conv_model.h
  EXPLICITLY_SPECIFIED_TEST_GENERATOR_INPUTS := \
  tensorflow/lite/micro/models/keyword_scrambled.tflite
  $(eval $(call microlite_test,memory_arena_threshold_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS), \
  $(EXPLICITLY_SPECIFIED_TEST_GENERATOR_INPUTS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/micro_allocator_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,micro_allocator_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

EXPLICITLY_SPECIFIED_TEST:= tensorflow/lite/micro/recording_micro_allocator_test.cc
ifneq ($(findstring $(EXPLICITLY_SPECIFIED_TEST),$(MICROLITE_TEST_SRCS)),)
  MICROLITE_TEST_SRCS := $(filter-out $(EXPLICITLY_SPECIFIED_TEST), $(MICROLITE_TEST_SRCS))
  EXPLICITLY_SPECIFIED_TEST_SRCS := \
  $(EXPLICITLY_SPECIFIED_TEST) \
  tensorflow/lite/micro/testing/test_conv_model.cc
  EXPLICITLY_SPECIFIED_TEST_HDRS := \
  tensorflow/lite/micro/testing/test_conv_model.h
  $(eval $(call microlite_test,recording_micro_allocator_test,\
  $(EXPLICITLY_SPECIFIED_TEST_SRCS),$(EXPLICITLY_SPECIFIED_TEST_HDRS)))
endif

# For all the tests that do not have any additional dependencies, we can
# add a make target in a common way.
$(foreach TEST_TARGET,$(filter-out tensorflow/lite/micro/kernels/%,$(MICROLITE_TEST_SRCS)),\
$(eval $(call microlite_test,$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))

ifeq ($(TARGET_SPECIFIC_MAKE_TEST),0)
test: $(MICROLITE_TEST_TARGETS)
integration_tests: $(MICROLITE_INTEGRATION_TEST_TARGETS)
endif

# Just build the test targets
build: $(MICROLITE_BUILD_TARGETS)

list_library_sources:
	@echo $(MICROLITE_CC_SRCS) $(MICROLITE_CC_KERNEL_SRCS)

list_library_headers:
	@echo $(MICROLITE_CC_HDRS)

list_third_party_sources:
	@echo $(THIRD_PARTY_CC_SRCS) $(THIRD_PARTY_KERNEL_CC_SRCS)

list_third_party_headers:
	@echo $(THIRD_PARTY_CC_HDRS)

list_generator_dir:
	@echo $(GENERATED_SRCS_DIR)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Removes third-party downloads.
clean_downloads:
	rm -rf $(MAKEFILE_DIR)/downloads

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d
.PRECIOUS: $(BINDIR)%_test

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
