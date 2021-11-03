/* libtpujesus.c
Copyright 2021 Shawn Presser

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// This is an example of building a custom "stub" libtpu.so library,
// with the ultimate goal of implementing your own "TPU" device for
// JAX.
//
// At a basic level, the overall goal is to do this:
//
//   gcc -shared -o libtpujesus.so libtpujesus.c
//   TPU_LIBRARY_PATH=libtpujesus.so TF_CPP_MIN_LOG_LEVEL=0 python3 -c 'import jax; print(jax.devices());'
//
// That will trick JAX into loading our own libtpu.so library.
//
// Unfortunately, unless you're running that on a Cloud TPU VM, you'll
// have to build your own jaxlib with TPU support. Luckily, that's
// not too hard. James Bradbury (https://twitter.com/jekbradbury)
// pointed out that we can simply pass --enable_tpu to build.py:
//
//   git clone https://github.com/google/jax ~/ml/jax-tpu
//   cd ~/ml/jax-tpu/build
//   python3 build.py --enable_tpu --bazel_options={--action_env=PATH,--remote_accept_cached=true,--spawn_strategy=standalone,--remote_local_fallback=false,--remote_timeout=600}
//
// (Those --bazel_options enable build caching, which is absolutely
// essential if you intend to modify the jaxlib tensorflow C++.
// Otherwise every code change will take 30 minutes instead of 20
// seconds. If you want to make custom C++ changes, see the
// section below for how to set that up.)
//
// then `pip3 install --force-reinstall` the resulting .whl file, e.g.
//
//   pip3 install --force-reinstall ~/ml/jax-tpu/build/dist/jaxlib-0.1.74-cp39-none-macosx_11_0_arm64.whl
//
// I usually install it in a venv, which I create with:
//
//   python3 -m virtualenv ~/ml/jax-tpu/venv -p python3 --system-site-packages
//   source ~/ml/jax-tpu/venv/bin/activate
//
// That way, your custom jaxlib doesn't override the system-installed one.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// How to make custom changes to tensorflow's C++ codebase (jaxlib):
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// First, modify ~/ml/jax-tpu/WORKSPACE as follows:
//
/*
# # To update TensorFlow to a new revision,
# # a) update URL and strip_prefix to the new git commit hash
# # b) get the sha256 hash of the commit by running:
# #    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
# #    and update the sha256 with the result.
# http_archive(
#     name = "org_tensorflow",
#     sha256 = "b2c8b912e7be71306ab6fee063fb4ec1dfbe7158e7e8469d674f8af6583434d4",
#     strip_prefix = "tensorflow-e98b052c08e5d1e7906ac2f6caf95c51a1e04985",
#     urls = [
#         "https://github.com/tensorflow/tensorflow/archive/e98b052c08e5d1e7906ac2f6caf95c51a1e04985.tar.gz",
#     ],
# )

# For development, one can use a local TF repository instead.
local_repository(
   name = "org_tensorflow",
   path = "tensorflow",
)
*/
//
// i.e. comment out the call to http_archive(), and add local_repository(name = "org_tensorflow", path="tensorflow").
//
// Then do:
//
//   git clone --recursive https://github.com/tensorflow/tensorflow ~/ml/tensorflow-tpu
//   ln -s $(HOME)/ml/tensorflow-tpu ~/ml/jax-tpu/tensorflow
//
// Finally, you can re-run the above build command.
//
// Putting it all together, here's what my process looks like:
//
//   cd ~/ml/jax-tpu/build
//   python3 build.py --enable_tpu --bazel_options={--action_env=PATH,--remote_accept_cached=true,--spawn_strategy=standalone,--remote_local_fallback=false,--remote_timeout=600}
//   python3 -m virtualenv ~/ml/jax-tpu/venv -p python3 --system-site-packages
//   source ~/ml/jax-tpu/venv/bin/activate
//   pip3 install --force-reinstall ~/ml/jax-tpu/build/dist/jaxlib-*.whl
//   gcc -shared -o ~/ml/libtpujesus/libtpujesus.{so,c}
//   TPU_LIBRARY_PATH=~/ml/libtpujesus/libtpujesus.so TF_CPP_MIN_LOG_LEVEL=0 python3 -c 'import jax; print(jax.devices());'
//
// And I end up getting output like this: https://gist.github.com/shawwn/2591c555ab918020d6be2ee121000c23

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Questions?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// DM me on twitter! I'm happy to talk: https://twitter.com/theshawwn
//
// Or you can roll the dice by emailing me at shawnpresser@gmail.com.
// I occasionally check it.

#include <Python.h>

static PyObject *my_callback = NULL;

#include <stdio.h>
#include <stdlib.h>

#if 0
// this is libtpu.h from tensorflow. Not strictly necessary yet.
#include "libtpu.h"
#else
typedef int bool;
#endif

//void TpuDriver_Initialize(struct TpuDriverFn* driver_fn, bool initialize) { printf("TpuDriver_Initialize\n"); }

void import_libtpu() {
  /* 1st: Import the module */
  PyObject *pylibtpu = PyImport_ImportModule("libtpu");
  if (!pylibtpu) {
    PyErr_Print();
    printf("libtpujesus: Error importing libtpu\n");
  }
#if 0
  PyObject* ModuleString = PyString_FromString((char*) "libtpu");
  if (!ModuleString) {
    PyErr_Print();
    printf("libtpujesus: Error importing libtpu\n");
  }

  PyObject* Module = PyImport_Import(ModuleString);
  if (!Module) {
    PyErr_Print();
    printf("libtpujesus: Error importing libtpu\n");
  }

  /* 2nd: Getting reference to the function */
  PyObject* Function = PyObject_GetAttrString(Module, (char*)"link_list");
  if (!Function) {
    PyErr_Print();
    printf("libtpujesus: Error getting link_list()\n");
  }
#endif
}

// tensorflow/core/tpu/tpu_api_dlsym_initializer.cc:64
void TfTpu_Initialize(bool init_library, int num_args, const char** args) {
  printf("TfTpu_Initialize(init_library=%d, num_args=%d)\n", init_library, num_args);
  for (int i = 0; i < num_args; i++) {
    printf("  args[%d] = \"%s\"\n", i, args[i]);
  }
  import_libtpu();
}


#include <inttypes.h>

typedef ssize_t ret_t;

#define STUB(x) ret_t x(ret_t arg1, ret_t arg2, ret_t arg3, ret_t arg4, ret_t arg5) { \
    void* p = NULL; \
    void* arg = (void*)&p; \
    void** argv = &p; \
    ret_t ret = 0; \
    printf("libtpujesus.so: " #x " called(%"PRId64",%"PRId64",%"PRId64",%"PRId64",%"PRId64")\n", arg1, arg2, arg3, arg4, arg5); \
    if (my_callback) { \
        printf("libtpujesus.so: " #x " building...\n"); \
        /* Time to call the callback */ \
        /*PyObject* arglist = Py_BuildValue("(s, L, L, L, L, L, L, L)", #x, argv, arg, arg1, arg2, arg3, arg4, arg5);*/ \
        PyObject* arglist = Py_BuildValue("(s, L, L, L, L, L, L, L)", #x, 0, 0, arg1, arg2, arg3, arg4, arg5); \
        printf("libtpujesus.so: " #x " calling...\n"); \
        PyObject* result = PyObject_CallObject(my_callback, arglist); \
        printf("libtpujesus.so: " #x " returned\n"); \
        Py_DECREF(arglist); \
        if (result) { \
          ret_t i = 0; \
          if (PyArg_ParseTuple(result, "L", &i)) { \
            ret = i; \
          } else { \
            printf("libtpujesus.so: " #x " FAILED TO PARSE RESULT\n"); \
          } \
          Py_DECREF(result); \
        } \
    } \
    printf("libtpujesus.so: returning %"PRId64" from " #x " callback\n", (long long)ret); \
    /*return Py_BuildValue("i", 42);*/ \
    return ret; \
}

//STUB(ConfigureDistributedTpuOp_DoWork)

#if 0
tensorflow::Status SetTpuOpsStructFns(void* library_handle) {
  // Constant cast so that we can initialize the functions. The functions are
  // mutable here because this is the only place where they are initialized.
  auto* ops_api_fn = const_cast<TfTpu_OpsApiFn*>(tensorflow::tpu::OpsApiFn());
#endif

#define TFTPU_SET_FN(api_fn, name) STUB(name)
  TFTPU_SET_FN(ops_api_fn, ConfigureDistributedTpuOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, WaitForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, InitializeHostForDistributedTpuOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, SetGlobalTPUArrayOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, DisconnectDistributedTpuChipsOp_DoWork);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_FreeCharArray);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_FreeInt32Array);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_HasTPUPodState);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_TpusPerHost);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_TpuMemoryLimit);
  TFTPU_SET_FN(ops_api_fn,
               TpuConfigurationApi_RemoteCompilationCacheSizeInBytes);
  TFTPU_SET_FN(ops_api_fn,
               TpuConfigurationApi_CompilationCacheServerAddressFromConfig);
  TFTPU_SET_FN(ops_api_fn, TpuConfigurationApi_GetServerAddressAndPort);

  TFTPU_SET_FN(ops_api_fn, TpuMeshState_Create);
  TFTPU_SET_FN(ops_api_fn, TpuMeshState_Free);
  TFTPU_SET_FN(ops_api_fn, TpuMeshState_MeshCommonState);

  TFTPU_SET_FN(ops_api_fn, TpuCompile_CompileAndBuild);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_XrtCompileAndBuild);

  TFTPU_SET_FN(ops_api_fn, TpuExecutable_LoadProgramAndEnqueueToStream);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_HostShapeToDeviceShape);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_ShapeSize);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_ShapeSizeCompact);
  TFTPU_SET_FN(ops_api_fn, HardwareLayout_ShapeSizeCompactRaw);

  TFTPU_SET_FN(ops_api_fn, TpuExecute_RuntimeInputToPaddedData);

  TFTPU_SET_FN(ops_api_fn, TpuProgram_New);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_Free);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_NewArray);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_FreeArray);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_UnloadAndDestroy);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetProgramSize);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_LogProgramMemorySummary);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetExecutableInfo);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetHostTransferInfo);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetHloMetadata);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetMayModifyVariables);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_HasSharding);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetTpuProgram);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_SerializeTpuExecutable);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_SerializeCompilerMetadata);
  TFTPU_SET_FN(ops_api_fn,
               TpuProgram_DeserializeFromGetTpuProgramResponseProto);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_GetFingerprint);
  TFTPU_SET_FN(ops_api_fn, TpuProgram_DestroyFingerprint);

  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_Create);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_Free);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_Initialize);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_StopChipHeartbeats);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_CloseTpuHost);
  TFTPU_SET_FN(ops_api_fn, TpuNodeContext_CompactionSupported);

  TFTPU_SET_FN(ops_api_fn, TpuTopology_AvailableCoreCount);
  TFTPU_SET_FN(ops_api_fn, TpuNetUtil_RecycleUnusedPort);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_IsTpuCompilationEnabled);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_CreateCompilationCacheKey);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_DestroyCompilationCacheKey);
  TFTPU_SET_FN(ops_api_fn, TpuCompile_CreateGuaranteedConstFingerprint);

  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Create);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Destroy);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Start);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_Stop);
  TFTPU_SET_FN(ops_api_fn, TpuProfiler_CollectData);

  TFTPU_SET_FN(ops_api_fn, TfTpu_InitializeTpuModelServer);

  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_Create);
  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_Destroy);
  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_GetOrdinal);
  TFTPU_SET_FN(ops_api_fn, TfTpuOrdinalSelector_DequeueFromCoreSelector);
  TFTPU_SET_FN(ops_api_fn, TfTpu_GetTpuPartitionedCallParams);
#undef TFTPU_SET_FN

#if 0
  return tensorflow::Status::OK();
}
#endif


#if 0
tensorflow::Status SetExecutorStructFn(void* library_handle) {
  auto* executor_fn = tensorflow::tpu::ExecutorApiFn();
#endif

#define TFTPU_SET_FN(api_fn, name) STUB(name)
  TFTPU_SET_FN(executor_fn, TpuPlatform_New);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Free);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Initialize);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Initialized);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetExecutor);
  TFTPU_SET_FN(executor_fn, TpuPlatform_Id);
  TFTPU_SET_FN(executor_fn, TpuPlatform_VisibleDeviceCount);
  TFTPU_SET_FN(executor_fn, TpuPlatform_TpuMemoryLimit);
  TFTPU_SET_FN(executor_fn, TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetTopologyPtr);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetHostLocation);
  TFTPU_SET_FN(executor_fn, TpuPlatform_GetRuntimeVersion);

  TFTPU_SET_FN(executor_fn, TpuExecutor_Init);
  TFTPU_SET_FN(executor_fn, TpuExecutor_Free);
  TFTPU_SET_FN(executor_fn, TpuExecutor_PlatformDeviceCount);
  TFTPU_SET_FN(executor_fn, TpuExecutor_Allocate);
  TFTPU_SET_FN(executor_fn, TpuExecutor_Deallocate);
  TFTPU_SET_FN(executor_fn, TpuExecutor_GetAllocatorStats);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeviceMemoryUsage);
  TFTPU_SET_FN(executor_fn, TpuExecutor_AllocateStream);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeallocateStream);
  TFTPU_SET_FN(executor_fn, TpuExecutor_CreateStreamDependency);
  TFTPU_SET_FN(executor_fn, TpuExecutor_GetStatus);
  TFTPU_SET_FN(executor_fn, TpuExecutor_GetCoreLocation);
  TFTPU_SET_FN(executor_fn, TpuExecutor_AllocateEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeallocateEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_PollForEventStatus);
  TFTPU_SET_FN(executor_fn, TpuExecutor_RecordEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_WaitForEvent);
  TFTPU_SET_FN(executor_fn, TpuExecutor_AllocateTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DeallocateTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_StartTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_StopTimer);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SynchronousMemcpyToHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SynchronousMemcpyFromHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_MemcpyToHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_MemcpyFromHost);
  TFTPU_SET_FN(executor_fn, TpuExecutor_EnqueueInfeed);
  TFTPU_SET_FN(executor_fn, TpuExecutor_DequeueOutfeed);
  TFTPU_SET_FN(executor_fn, TpuExecutor_WaitForInfeedReady);
  TFTPU_SET_FN(executor_fn, TpuExecutor_WaitForOutfeedReady);
  TFTPU_SET_FN(executor_fn, TpuExecutor_BlockHostUntilDone);
  TFTPU_SET_FN(executor_fn, TpuExecutor_BlockUntilDoneOrFailed);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SyncAndForgetFailedStreams);
  TFTPU_SET_FN(executor_fn, TpuExecutor_SynchronizeAllActivity);
  TFTPU_SET_FN(executor_fn, TpuExecutor_UnloadAllPrograms);
  TFTPU_SET_FN(executor_fn, TpuExecutor_EnqueueCompactionOnStreamForHbm);

  TFTPU_SET_FN(executor_fn, TpuStream_New);
  TFTPU_SET_FN(executor_fn, TpuStream_Free);
  TFTPU_SET_FN(executor_fn, TpuStream_Stream);
  TFTPU_SET_FN(executor_fn, TpuStream_Status);
  TFTPU_SET_FN(executor_fn, TpuStream_IsSameSharedMemoryLocation);
  TFTPU_SET_FN(executor_fn, TpuStream_EnqueueTransferHostToDevice);
  TFTPU_SET_FN(executor_fn, TpuStream_EnqueueTransferDeviceToHost);
  TFTPU_SET_FN(executor_fn, TpuStream_TpuEnqueueOnDeviceSendRecvLocal);

  TFTPU_SET_FN(executor_fn, TpuEvent_New);
  TFTPU_SET_FN(executor_fn, TpuEvent_Free);

  TFTPU_SET_FN(executor_fn, TpuTimer_New);
  TFTPU_SET_FN(executor_fn, TpuTimer_Free);
  TFTPU_SET_FN(executor_fn, TpuTimer_Nanoseconds);
  TFTPU_SET_FN(executor_fn, TpuTimer_Microseconds);

  TFTPU_SET_FN(executor_fn, TpuStatus_New);
  TFTPU_SET_FN(executor_fn, TpuStatus_Create);
  TFTPU_SET_FN(executor_fn, TpuStatus_Set);
  TFTPU_SET_FN(executor_fn, TpuStatus_Free);
  TFTPU_SET_FN(executor_fn, TpuStatus_Message);
  TFTPU_SET_FN(executor_fn, TpuStatus_Code);
  TFTPU_SET_FN(executor_fn, TpuStatus_Ok);

  TFTPU_SET_FN(executor_fn, TpuStreamExecutorConfig_Default);
  TFTPU_SET_FN(executor_fn, TpuStreamExecutorConfig_SetOrdinal);
  TFTPU_SET_FN(executor_fn, TpuStreamExecutorConfig_Free);

  TFTPU_SET_FN(executor_fn, TpuDeviceDescription_New);
  TFTPU_SET_FN(executor_fn, TpuDeviceDescription_Free);

  TFTPU_SET_FN(executor_fn, TpuExecutor_CreateDeviceDescription);
  TFTPU_SET_FN(executor_fn, TpuExecutor_NewDeviceOptions);
  TFTPU_SET_FN(executor_fn, TpuExecutor_FreeDeviceOptions);
  TFTPU_SET_FN(executor_fn, TpuExecutor_HostCallback);

  TFTPU_SET_FN(executor_fn, TpuTransferManager_New);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_Free);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_PlatformId);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_HostShapeToDeviceShape);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralToDeviceAsync);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralFromDevice);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_GetByteSizeRequirement);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_ChooseCompactLayoutForShape);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_CanShapedBufferBeAccessedNow);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_CanBufferBeAccessedNow);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_WriteSingleTupleIndexTable);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_GetInfeedLayout);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_LinearizeToBuffers);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_FreeBuffers);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralToInfeed);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferBuffersToInfeed);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_TransferLiteralFromOutfeed);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_ResetDevices);
  TFTPU_SET_FN(executor_fn, TpuTransferManager_ReadDynamicShapes);

  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_New);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_Free);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_AssignDevices);
  TFTPU_SET_FN(executor_fn, TpuComputationPlacer_AssignLocalDevices);

  TFTPU_SET_FN(executor_fn, TpuTopology_LogicalDevicesPerHost);
  TFTPU_SET_FN(executor_fn, TpuTopology_LogicalDevicesPerChip);
  TFTPU_SET_FN(executor_fn, TpuTopology_HostCount);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipsPerHost);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipBounds_X);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipBounds_Y);
  TFTPU_SET_FN(executor_fn, TpuTopology_ChipBounds_Z);
  TFTPU_SET_FN(executor_fn, TpuTopology_HasChip);
  TFTPU_SET_FN(executor_fn, TpuTopology_CoreForId);
  TFTPU_SET_FN(executor_fn, TpuTopology_Core);
  TFTPU_SET_FN(executor_fn, TpuTopology_NumCores);
  TFTPU_SET_FN(executor_fn, TpuTopology_Cores);
  TFTPU_SET_FN(executor_fn, TpuTopology_IdForHost);
  TFTPU_SET_FN(executor_fn, TpuTopology_Version);

  TFTPU_SET_FN(executor_fn, TpuCoreLocation_ChipCoordinates);
  TFTPU_SET_FN(executor_fn, TpuCoreLocation_HostCoordinates);
  TFTPU_SET_FN(executor_fn, TpuCoreLocation_Index);
  TFTPU_SET_FN(executor_fn, TpuCoreLocation_Id);

  TFTPU_SET_FN(executor_fn, TpuHostLocation_Id);
  TFTPU_SET_FN(executor_fn, TpuHostLocation_NumCores);
  TFTPU_SET_FN(executor_fn, TpuHostLocation_Cores);

  TFTPU_SET_FN(executor_fn, TpuCompiler_New);
  TFTPU_SET_FN(executor_fn, TpuCompiler_Free);

  TFTPU_SET_FN(executor_fn, TpuCompiler_RunHloPasses);
  TFTPU_SET_FN(executor_fn, TpuCompiler_RunBackend);
  TFTPU_SET_FN(executor_fn, TpuCompiler_Compile);
  TFTPU_SET_FN(executor_fn, TpuCompiler_ShapeSize);
  TFTPU_SET_FN(executor_fn, TpuExecutable_ExecuteAsyncOnStream);
  TFTPU_SET_FN(executor_fn, TpuExecutable_FreeXlaShapeIndexArray);
  TFTPU_SET_FN(executor_fn, TpuExecutable_FreeMaybeOwningDeviceMemoryArray);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Fingerprint);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Serialize);
  TFTPU_SET_FN(executor_fn, TpuExecutableSerialize_GetByteSize);
  TFTPU_SET_FN(executor_fn, TpuExecutableSerialize_WriteToArray);
  TFTPU_SET_FN(executor_fn, TpuExecutableSerialize_FreeHandle);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Deserialize);
  TFTPU_SET_FN(executor_fn, TpuExecutable_HloModule);
  TFTPU_SET_FN(executor_fn, TpuExecutable_Free);

  TFTPU_SET_FN(executor_fn, XlaShapeToTpuShapeRepresentation);
  TFTPU_SET_FN(executor_fn, XlaShapeToTpuPaddedShape);
#undef TFTPU_SET_FN

#if 0
  return tensorflow::Status::OK();
}
#endif

static PyObject *
get_answer(PyObject *self, PyObject *args)
{
    if (my_callback) {
        /* Time to call the callback */
        int arg = 123;
        PyObject* arglist = Py_BuildValue("(i)", arg);
        PyObject* result = PyObject_CallObject(my_callback, arglist);
        if (result) {
          Py_DECREF(result);
        }
        Py_DECREF(arglist);
    }
    return Py_BuildValue("i", 42);
}

static int
PyLibTpuJesus_System(const char *command)
{
    return system(command);
}

static PyObject *
libtpujesus_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = PyLibTpuJesus_System(command);
    return PyLong_FromLong(sts);
}

static PyObject *
libtpujesus_free(PyObject *self, PyObject *args)
{
    Py_ssize_t addr = 0;
    if (!PyArg_ParseTuple(args, "n", &addr))
        return NULL;
    void* p = (void*)addr;
    if (p) {
      free(p);
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
libtpujesus_malloc(PyObject *self, PyObject *args)
{
    Py_ssize_t nbytes = 0;
    if (!PyArg_ParseTuple(args, "n", &nbytes))
        return NULL;
    void* p = nbytes >= 0 ? malloc(nbytes) : (void*)0;
    return PyLong_FromVoidPtr(p);
}

static PyObject *
libtpujesus_set_callback(PyObject *self, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *temp;

    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            /*
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
            */
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(my_callback);  /* Dispose of previous callback */
        my_callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}

static PyMethodDef LibTpuJesusMethods[] = {
    {"get_answer",  get_answer, METH_VARARGS, "The meaning of life."},
    {"system",  libtpujesus_system, METH_VARARGS, "Execute a shell command."},
    {"set_callback",  libtpujesus_set_callback, METH_VARARGS, "Set the libtpu callback."},
    {"free",  libtpujesus_free, METH_VARARGS, "free(ptr)"},
    {"malloc",  libtpujesus_malloc, METH_VARARGS, "malloc(nbytes)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libtpujesusmodule = {
    PyModuleDef_HEAD_INIT,
    "libtpujesus",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    LibTpuJesusMethods
};

static PyObject *SpamError;

PyMODINIT_FUNC
PyInit_libtpujesus(void) {
    PyObject *m;

    m = PyModule_Create(&libtpujesusmodule);
    if (m == NULL)
        return NULL;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0) {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }
    import_libtpu();

    return m;
}