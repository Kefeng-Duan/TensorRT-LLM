# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
include(FetchContent)

set(SRCS
    utils/mpiUtils.cpp
    utils/numpyUtils.cpp
    utils/runtimeUtils.cpp
    utils/debugUtils.cu
    utils/speculativeChoicesUtils.cpp
    bufferManager.cpp
    cudaMemPool.cpp
    decodingLayerWorkspace.cpp
    eagleBuffers.cpp
    explicitDraftTokensBuffers.cpp
    lookaheadBuffers.cpp
    layerProfiler.cpp
    loraManager.cpp
    loraUtils.cpp
    loraModule.cpp
    loraCache.cpp
    decodingOutput.cpp
    decoderState.cpp
    gptDecoder.cpp
    gptDecoderBatched.cpp
    gptJsonConfig.cpp
    iBuffer.cpp
    iTensor.cpp
    ipcUtils.cpp
    ipcSocket.cpp
    ipcNvlsMemory.cpp
    memoryCounters.cpp
    ncclCommunicator.cpp
    promptTuningParams.cpp
    runtimeKernels.cu
    tllmBuffers.cpp
    tllmRuntime.cpp
    tllmStreamReaders.cpp
    tllmLogger.cpp
    workerPool.cpp
    worldConfig.cpp)

include_directories(${API_INCLUDE_DIR}/tensorrt_llm/runtime)

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(runtime_src OBJECT ${SRCS})
set_property(TARGET runtime_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET runtime_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(runtime_src PRIVATE ${MPI_C_INCLUDE_DIRS})

if(ENABLE_MULTI_DEVICE)
  target_link_libraries(runtime_src PUBLIC ${NCCL_LIB})
endif()
