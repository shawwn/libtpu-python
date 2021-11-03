from __future__ import annotations
import sys
__saved_stdout__, sys.stdout = sys.stdout, sys.__stdout__
import libtpujesus
# import libtpu
from typing import NamedTuple
import ctypes
import inspect
import traceback
import pdb
import posix
import builtins
import copy
from dataclasses import dataclass, is_dataclass
from typing import Tuple, Sequence, List, ClassVar, Any
from enum import Enum, auto
from pyembc import pyembc_struct, pyembc_union
from pprint import pprint as pp
from functools import partial

struct = partial(pyembc_struct, pack=8)
union = partial(pyembc_union, pack=8)

def malloc(sz):
  if hasattr(sz, 'value'):
    sz = sz.value
  if isinstance(sz, int) and sz >= 0:
    return ctypes.cast(libtpujesus.malloc(sz), void_p)
  else:
    panic("malloc: invalid input", sz)

def free(p):
  if p is not None:
    if hasattr(p, 'value'):
      p = p.value
    if p is not None and isinstance(p, int) and p != 0:
      libtpujesus.free(p)
    else:
      panic("free: invalid input", p)

def is_dataclass_instance(obj):
  return is_dataclass(obj) and not isinstance(obj, type)

class Wrappable:
  pass

def warn(msg, *args):
  print(msg, args, file=sys.stderr)

def exit(code=0):
  posix._exit(code)

def fatal(msg, *args):
  warn(msg, *args)
  exit()

def brk():
  mypdb = pdb.Pdb(stdout=sys.__stderr__)
  mypdb.reset()
  mypdb.set_trace()

def pm(t=None):
  mypdb = pdb.Pdb(stdout=sys.__stderr__)
  mypdb.reset()
  if t is None:
    if hasattr(sys, 'last_traceback'):
      t = sys.last_traceback
    else:
      t = sys.exc_info()[2]
  mypdb.interaction(None, t)


def panic(msg, *args):
  warn(msg, *args)
  brk()
  exit()

stackptr_t = ctypes.POINTER(ctypes.c_ssize_t)
int8_t = ctypes.c_int8
int16_t = ctypes.c_int16
int32_t = ctypes.c_int32
int64_t = ctypes.c_int64
uint8_t = ctypes.c_uint8
uint16_t = ctypes.c_uint16
uint32_t = ctypes.c_uint32
uint64_t = ctypes.c_uint64
float32_t = ctypes.c_float
float64_t = ctypes.c_double

int64_p = ctypes.POINTER(ctypes.c_int64)

size_t = ctypes.c_size_t
ssize_t = ctypes.c_ssize_t

cstr_t = ctypes.c_char_p # const char*
cstr_array_t = ctypes.POINTER(cstr_t) # const char**

int_t = int32_t
float_t = float32_t
bool_t = ctypes.c_bool

ptr_t = void_p = ctypes.c_void_p
int_out = int_p = ctypes.POINTER(int_t)
ptr_out = ptr_p = ctypes.POINTER(ptr_t)

def is_cpointertype(x): return issubclass(x, ctypes._Pointer) or is_cdatatype(x) and x.__name__.endswith("_p") # a hack
def is_cdatatype(x): return issubclass(x, ctypes._SimpleCData)
def is_cpointer(x): return is_cpointertype(type(x))
def is_cstructtype(x): return issubclass(x, ctypes.Structure)
def is_cstruct(x): return isinstance(x, ctypes.Structure)
def is_cdata(x): return is_cdatatype(type(x))
def is_nullptr(x): x = getattr(x, 'value', x); return x is None or x == 0

def cvalue(x):
  if not is_cdata(x):
    panic("Not cdata", x)
  return x.value

def deref(x):
  if is_nullptr(x):
    panic("Tried to deref a null pointer", x)
  if isinstance(x, int):
    if x in objs:
      return objs[x]
    panic("Tried to deref unknown address", x)
  if is_cpointer(x):
    if hasattr(x, 'contents'):
      return x[0]
  panic("Can't deref", x)

objs = {}
stack_ptr = None

def addr(x):
  try:
    return ctypes.addressof(x)
  except TypeError:
    return id(x)

def pin(x, ptr = None):
  if ptr is None:
    ptr = addr(x)
  objs[ptr] = x
  return ptr

def new(x):
  return pin(x)

def delete(x):
  if addr(x) in objs:
    del objs[addr(x)]
  elif x:
    panic('Delete of unknown ptr', x)

_argv = []

def arg(i) -> ctypes.c_ssize_t:
  #assert stack_ptr
  #return stack_ptr.contents[i]
  return _argv[i]

def argv(f, i):
  val = arg(i)
  params = list(inspect.signature(f).parameters.values())
  kind = params[i].annotation
  if isinstance(kind, str):
    kind = getattr(builtins, kind, kind)
  if isinstance(kind, str):
    kind = globals().get(kind)
  assert kind is not None
  assert not isinstance(kind, str)
  if kind == int:
    # return ctypes.cast(val, int_t)
    #return ctypes.cast((int_t * 1)(val), ctypes.POINTER(int_t))[0]
    return (int_t * 1)(val)[0]
  if kind == bool:
    return val == 1
  if val == 0:
    return None
  if kind is cstr_t and isinstance(val, str):
    return val
  if is_cpointertype(kind):
    return ctypes.cast(val, kind)
  if is_cstructtype(kind):
    return ctypes.cast(val, ctypes.POINTER(kind)).contents # TODO: this is dubious
  if is_cdatatype(kind):
    #return ctypes.cast(val, kind)
    return kind(val)
  if val in objs:
    o = objs[val]
    if isinstance(o, kind):
      return o
    else:
      panic('object was an unrelated type', dict(expected=kind, actual=type(o)), o)
  if isinstance(val, TpuType):
    warn('untracked val', val)
    return val
  panic('Unknown annotation', kind, f, i)

class NewFree:
  @classmethod
  def New(cls):
    return cls.__wraps__()
  def Free(self):
    return delete(self)

def getmembers(cls):
  return [(k, v) for k, v in inspect.getmembers(cls)
          if isinstance(k, str) and not (k.startswith('__') and k.endswith('__'))]

class TpuType:
  def __init_subclass__(cls, use_name=None, wraps=None, **kwargs):
    super().__init_subclass__(**kwargs)
    if use_name is None:
      use_name = cls.__name__
    print('init_subclass', wraps, cls, kwargs, file=sys.stderr)
    if wraps is not None:
      cls.__wraps__ = wraps
      # print('TKTK', wraps, file=sys.stderr)
    for name, impl in getmembers(cls):
      if len(name) > 1 and not (name[0].isupper() and not name[1].isupper()):
        warn(f'expected FooBarBaz style name, got {name}')
      else:
        global_name = f'{use_name}_{name}'
        assert global_name not in globals(), f'{global_name} already implemented'
        globals()[global_name] = impl
        print('IMPLEMENTING:', global_name, impl, file=sys.stderr)



@dataclass
class SE_TpuTopology_Core(TpuType, use_name='TpuCoreLocation'):
  x: int = 0
  y: int = 0
  z: int = 0
  id: int = 0
  index: int = 0
  # void TpuCoreLocation_ChipCoordinates(SE_TpuTopology_Core* tpu_core_location,
  #                                      int* x, int* y, int* z);
  def ChipCoordinates(self: SE_TpuTopology_Core,
                      x: int_out, y: int_out, z: int_out):
    x[0] = self.x if self is not None else 0
    y[0] = self.y if self is not None else 0
    z[0] = self.z if self is not None else 0
  # void TpuCoreLocation_HostCoordinates(SE_TpuTopology_Core* tpu_core_location,
  #                                      int* x, int* y, int* z);
  def HostCoordinates(self: SE_TpuTopology_Core,
                      x: int_out, y: int_out, z: int_out):
    x[0] = self.x if self is not None else 0
    y[0] = self.y if self is not None else 0
    z[0] = self.z if self is not None else 0
  # int TpuCoreLocation_Index(SE_TpuTopology_Core* tpu_core_location);
  def Index(self: SE_TpuTopology_Core) -> int_t: return self.index if self else 0
  # int TpuCoreLocation_Id(SE_TpuTopology_Core* tpu_core_location);
  def Id(self: SE_TpuTopology_Core) -> int_t: return self.id if self else 0

#SE_TpuTopology_Core_array = ctypes.POINTER(ctypes.POINTER(ctypes.py_object))
SE_TpuTopology_Core_array = ptr_out

# enum TpuCoreTypeEnum {
#   kTensorCore,
#   kEmbeddingV1,
#   kEmbeddingV2,
# };
TpuCoreTypeEnum = int

# enum TpuVersionEnum {
#   kUnknownTpuVersion,
#   kTpuV2,
#   kTpuV3,
#   kTpuV4,
# };
class TpuVersionEnum(Enum):
  kUnknownTpuVersion = auto()
  kTpuV2 = auto()
  kTpuV3 = auto()
  kTpuV4 = auto()

# typedef struct TpuRuntimeVersion {
#   // The three version numbers are: major, minor, patch
#   int version[3];
#   const char* metadata;
#   size_t metadata_size;
# } TpuRuntimeVersion;

int3_t = int_t * 3

@struct
class TpuRuntimeVersion:
  # // The three version numbers are: major, minor, patch
  # int version[3];
  version: int3_t
  # const char* metadata;
  metadata: cstr_t
  # size_t metadata_size;
  metadata_size: size_t
  # def __init__(self):
  #   super().__init__()
  #   for name, ctype in self._fields_:
  #     if ctype == cstr_t:
  #       setattr(self, name, ctype(b""))
  # _fields_: ClassVar[Sequence[str, Any]] = [
  #   #   // The three version numbers are: major, minor, patch
  #   #   int version[3];
  #   ('version', int_t * 3),
  #   #   const char* metadata;
  #   ('metadata', cstr_t),
  #   #   size_t metadata_size;
  #   ('metadata_size', size_t),
  # ]

@dataclass
class SE_TpuTopology_Host(TpuType, use_name='TpuHostLocation'):
  id: int
  cores: List[SE_TpuTopology_Core]
  # int TpuHostLocation_Id(SE_TpuTopology_Host* tpu_host_location);
  def Id(self: SE_TpuTopology_Host) -> int_t:
    return self.id if self is not None else 0
  # int TpuHostLocation_NumCores(SE_TpuTopology_Host* tpu_host_location,
  #                              TpuCoreTypeEnum tpu_core_type);
  def NumCores(self: SE_TpuTopology_Host, tpu_core_type: TpuCoreTypeEnum) -> int_t:
    return len(self.cores) if self is not None else 0
  # // 'cores' should be a preallocated array of size TpuHostLocation_NumCores.
  # void TpuHostLocation_Cores(SE_TpuTopology_Host* tpu_host_location,
  #                            TpuCoreTypeEnum tpu_core_type,
  #                            SE_TpuTopology_Core** cores);
  def Cores(self: SE_TpuTopology_Host,
            tpu_core_type: TpuCoreTypeEnum,
            cores: SE_TpuTopology_Core_array):
    if self is not None:
      for i, core in enumerate(self.cores):
        cores[i] = pin(core)

@dataclass
class SE_TpuTopology(TpuType, use_name='TpuTopology'):
  id: int
  cores: List[SE_TpuTopology_Core]
  # int TpuTopology_LogicalDevicesPerHost(SE_TpuTopology* tpu_topology,
  #                                       TpuCoreTypeEnum tpu_core_type);
  # int TpuTopology_LogicalDevicesPerChip(SE_TpuTopology* tpu_topology,
  #                                       TpuCoreTypeEnum tpu_core_type);
  # int TpuTopology_HostCount(SE_TpuTopology* tpu_topology);
  # int TpuTopology_ChipsPerHost(SE_TpuTopology* tpu_topology);
  #
  # int TpuTopology_ChipBounds_X(SE_TpuTopology* tpu_topology);
  # int TpuTopology_ChipBounds_Y(SE_TpuTopology* tpu_topology);
  # int TpuTopology_ChipBounds_Z(SE_TpuTopology* tpu_topology);
  # bool TpuTopology_HasChip(SE_TpuTopology* tpu_topology, int x, int y, int z);
  # SE_TpuTopology_Core* TpuTopology_CoreForId(SE_TpuTopology* tpu_topology,
  #                                            TpuCoreTypeEnum tpu_core_type,
  #                                            int id);
  # SE_TpuTopology_Core* TpuTopology_Core(SE_TpuTopology* tpu_topology,
  #                                       TpuCoreTypeEnum tpu_core_type, int x,
  #                                       int y, int z, int index);
  # int TpuTopology_NumCores(SE_TpuTopology* tpu_topology,
  #                          TpuCoreTypeEnum tpu_core_type);
  def NumCores(self: SE_TpuTopology, tpu_core_type: TpuCoreTypeEnum):
    return len(self.cores)
  # // 'cores' should be a preallocated array of size TpuTopology_NumCores.
  # void TpuTopology_Cores(SE_TpuTopology* tpu_topology,
  #                        TpuCoreTypeEnum tpu_core_type,
  #                        SE_TpuTopology_Core** cores);
  # int TpuTopology_IdForHost(SE_TpuTopology* tpu_topology, int x, int y, int z);
  def IdForHost(self: SE_TpuTopology, x: int, y: int, z: int) -> int:
    print('IdForHost', x, y, z)
    return 0
  # TpuVersionEnum TpuTopology_Version(SE_TpuTopology* tpu_topology);
  def Version(self: SE_TpuTopology) -> TpuVersionEnum:
    return TpuVersionEnum.kTpuV2.value
  # // 'cores' should be a preallocated array of size TpuTopology_NumCores.
  # void TpuTopology_Cores(SE_TpuTopology* tpu_topology,
  #                        TpuCoreTypeEnum tpu_core_type,
  #                        SE_TpuTopology_Core** cores);
  def Cores(self: SE_TpuTopology,
            tpu_core_type: TpuCoreTypeEnum,
            cores: SE_TpuTopology_Core_array, # SE_TpuTopology_Core**
            ):
    for i, core in enumerate(self.cores):
      cores[i] = pin(core)


#
# TpuPlatform / SE_Platform
#

@dataclass
class SE_Platform(TpuType, use_name='TpuPlatform'):
  inst: ClassVar[SE_Platform] = None
  @classmethod
  def get(cls) -> SE_Platform:
    return cls.inst
  devices: List
  executor: SE_StreamExecutor
  topology: SE_TpuTopology
  topology_host: SE_TpuTopology_Host
  runtime_version: TpuRuntimeVersion
  initialized: bool = False
  # def TpuPlatform_New() -> SE_Platform: return SE_Platform([None])
  @classmethod
  def New(cls: SE_Platform) -> SE_Platform:
    if cls.inst is None:
      cores = [SE_TpuTopology_Core()]
      cls.inst = cls(
        devices=[None],
        executor=SE_StreamExecutor(),
        topology=SE_TpuTopology(id=0, cores=cores),
        topology_host=SE_TpuTopology_Host(id=0, cores=cores),
        runtime_version=TpuRuntimeVersion(int3_t(1, 2, 3), cstr_t(b'foo'), size_t(3)),
      )
    return cls.inst
  # def TpuPlatform_Free(platform: SE_Platform): return delete(platform)
  def Free(self: SE_Platform):
    return delete(self)
  # void TpuPlatform_Initialize(SE_Platform* platform, size_t options_size,
  #                             const char** options_key,
  #                             const char** options_value, TF_Status* status);
  def Initialize(self: SE_Platform,
                 options_size: int,
                 options_key: cstr_array_t,
                 options_value: cstr_array_t,
                 status: TF_Status):
    assert self
    self.initialized = True
  # bool TpuPlatform_Initialized(SE_Platform* platform);
  def Initialized(self: SE_Platform) -> bool:
    return self and self.initialized
  # SE_StreamExecutor* TpuPlatform_GetExecutor(SE_Platform* platform,
  #                                            SE_StreamExecutorConfig* config,
  #                                            TF_Status* status);
  def GetExecutor(self: SE_Platform,
                  config: SE_StreamExecutorConfig,
                  status: TF_Status) -> SE_StreamExecutor:
    status.ok()
    return SE_StreamExecutor()
  # SE_PlatformId TpuPlatform_Id(SE_Platform* platform);
  # int64_t TpuPlatform_VisibleDeviceCount(SE_Platform* platform);
  def VisibleDeviceCount(self: SE_Platform) -> int64_t:
    return int64_t(len(self.devices))
  # int64_t TpuPlatform_TpuMemoryLimit(SE_Platform* platform);
  # bool TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy(SE_Platform* platform);
  # SE_TpuTopology* TpuPlatform_GetTopologyPtr(SE_Platform* platform);
  def GetTopologyPtr(self: SE_Platform) -> SE_TpuTopology:
    print('GetTopoPtr TKTK')
    return self.topology
  # SE_TpuTopology_Host* TpuPlatform_GetHostLocation(SE_Platform* platform);
  def GetHostLocation(self: SE_Platform) -> SE_TpuTopology_Host:
    print('GetHost TKTK')
    return self.topology_host
  # TpuRuntimeVersion TpuPlatform_GetRuntimeVersion(SE_Platform* platform);
  def GetRuntimeVersion(self: SE_Platform) -> TpuRuntimeVersion:
    return [self.runtime_version]

# TpuStatus / TF_Status
#

@dataclass
class TF_Status:
  code: int
  message: str
  @classmethod
  def copy(cls, other):
    assert isinstance(other, cls)
    return copy.copy(other)
  def set(self, other):
    self.code = other.code
    self.message = other.message
    return self
  def ok(self):
    self.set(self.OK)

TF_Status.OK = TF_Status(code=0, message="OK")

def TpuStatus_New() -> TF_Status: return TF_Status.copy(TF_Status.OK)
def TpuStatus_Create(code: int, msg: str) -> TF_Status: return TF_Status(code, msg)
def TpuStatus_Set(status: TF_Status, code: int32_t, msg: cstr_t, len: int = -1) -> None:
  status.code = code
  status.message = msg.value
def TpuStatus_Free(status: TF_Status): return delete(status)
def TpuStatus_Message(status: TF_Status): return status.message
def TpuStatus_Code(status: TF_Status): return status.code
def TpuStatus_Ok(status: TF_Status): return status.code == TF_Status.OK.code

#
# TpuStreamExecutorConfig / SE_StreamExecutorConfig
#
@dataclass
class SE_StreamExecutorConfig(TpuType, use_name='TpuStreamExecutorConfig'):
  inst: ClassVar = None
  ordinal: int32_t = -1
  @classmethod
  def Default(cls):
    if cls.inst is None:
      cls.inst = SE_StreamExecutorConfig(ordinal=0)
    return cls.inst


# SE_StreamExecutorConfig* TpuStreamExecutorConfig_Default();
def TpuStreamExecutorConfig_Default() -> SE_StreamExecutorConfig:
  return SE_StreamExecutorConfig()
# void TpuStreamExecutorConfig_SetOrdinal(SE_StreamExecutorConfig*, int ordinal);
def TpuStreamExecutorConfig_SetOrdinal(self: SE_StreamExecutorConfig, ordinal: int32_t):
  print('TpuStreamExecutorConfig_SetOrdinal', ordinal)
  self.ordinal = ordinal
# void TpuStreamExecutorConfig_Free(SE_StreamExecutorConfig*);
def TpuStreamExecutorConfig_Free(self: SE_StreamExecutorConfig):
  return delete(self)

#
# Tpu_Compiler
#
@dataclass
class Tpu_Compiler:
  pass

# // C API for XLA::Compiler interface
#
# TFTPU_CAPI_EXPORT Tpu_Compiler* TpuCompiler_New();
def TpuCompiler_New() -> Tpu_Compiler: return Tpu_Compiler()
# TFTPU_CAPI_EXPORT void TpuCompiler_Free(Tpu_Compiler* compiler);
def TpuCompiler_Free(compiler: Tpu_Compiler): return delete(compiler)



#
# TpuDeviceDescription / SE_DeviceDescription
#
class SE_DeviceDescription(ctypes.Structure):
    def __init__(self, *args, **kws):
      super().__init__(*args, **kws)
      print('SE_DeviceDescription_ctor', *args, kws)
      for name, ctype in self._fields_:
        if ctype == cstr_t:
          setattr(self, name, ctype(b""))

    _fields_ = [
      #   char* device_vendor;
      ('device_vendor', cstr_t),
      #   char* platform_version;
      ('platform_version', cstr_t),
      #   char* driver_version;
      ('driver_version', cstr_t),
      #   char* runtime_version;
      ('runtime_version', cstr_t),
      #   char* pci_bus_id;
      ('pci_bus_id', cstr_t),
      #   char* name;
      ('name', cstr_t),
      #
      #   int64_t thread_dim_limit_x;
      ('thread_dim_limit_x', int64_t),
      #   int64_t thread_dim_limit_y;
      ('thread_dim_limit_y', int64_t),
      #   int64_t thread_dim_limit_z;
      ('thread_dim_limit_z', int64_t),
      #   int64_t block_dim_limit_x;
      ('block_dim_limit_x', int64_t),
      #   int64_t block_dim_limit_y;
      ('block_dim_limit_y', int64_t),
      #   int64_t block_dim_limit_z;
      ('block_dim_limit_z', int64_t),
      #
      #   int64_t threads_per_core_limit;
      ('threads_per_core_limit', int64_t),
      #   int64_t threads_per_block_limit;
      ('threads_per_block_limit', int64_t),
      #   int64_t threads_per_warp;
      ('threads_per_warp', int64_t),
      #
      #   int64_t registers_per_core_limit;
      ('registers_per_core_limit', int64_t),
      #   int64_t registers_per_block_limit;
      ('registers_per_block_limit', int64_t),
      #
      #   int64_t device_address_bits;
      ('device_address_bits', int64_t),
      #   int64_t device_memory_size;
      ('device_memory_size', int64_t),
      #   int64_t memory_bandwidth;
      ('memory_bandwidth', int64_t),
      #
      #   int64_t shared_memory_per_core;
      ('shared_memory_per_core', int64_t),
      #   int64_t shared_memory_per_block;
      ('shared_memory_per_block', int64_t),
      #
      #   float clock_rate_ghz;
      ('clock_rate_ghz', float_t),
      #
      #   int cuda_compute_capability_major;
      ('cuda_compute_capability_major', int_t),
      #   int cuda_compute_capability_minor;
      ('cuda_compute_capability_minor', int_t),
      #
      #   int rocm_amdgpu_isa_version;
      ('rocm_amdgpu_isa_version', int_t),
      #   char* rocm_amdgpu_gcn_arch_name;
      ('rocm_amdgpu_gcn_arch_name', cstr_t),
      #
      #   int numa_node;
      ('numa_node', int_t),
      #   int core_count;
      ('core_count', int_t),
      #   bool ecc_enabled;
      ('ecc_enabled', bool_t),
    ]


# SE_DeviceDescription* TpuDeviceDescription_New();
def TpuDeviceDescription_New() -> SE_DeviceDescription: return SE_DeviceDescription()
# void TpuDeviceDescription_Free(SE_DeviceDescription* description);
def TpuDeviceDescription_Free(description: SE_DeviceDescription): return delete(description)


# typedef struct SE_DeviceMemoryBase {
#   void* opaque;
#   uint64_t size;
#   uint64_t payload;
# } SE_DeviceMemoryBase;

@struct
class SE_DeviceMemoryBase:
  opaque: void_p
  size: uint64_t
  payload: uint64_t

SE_DeviceMemoryBase_p = ctypes.POINTER(SE_DeviceMemoryBase)

#
# TpuExecutor / SE_StreamExecutor
#
@dataclass
class SE_StreamExecutor(TpuType, use_name='TpuExecutor'):
  # void TpuExecutor_CreateDeviceDescription(SE_StreamExecutor* executor,
  #                                          SE_DeviceDescription* description,
  #                                          TF_Status* status);
  def CreateDeviceDescription(self: SE_StreamExecutor,
                              description: SE_DeviceDescription,
                              status: TF_Status):
    status.ok()

  # void TpuExecutor_Init(SE_StreamExecutor* executor, int device_ordinal,
  #                       SE_DeviceOptions* device_options, TF_Status* status);
  # void TpuExecutor_Free(SE_StreamExecutor* executor);
  #
  # int TpuExecutor_PlatformDeviceCount(SE_StreamExecutor* executor);
  #
  # # SE_DeviceMemoryBase TpuExecutor_Allocate(SE_StreamExecutor* executor,
  # #                                          uint64_t size, int64_t memory_space);
  # def Allocate(self: SE_StreamExecutor,
  #              size: uint64_t, memory_space: int64_t) -> SE_DeviceMemoryBase:
  #   print('Allocate', dict(size=size, memory_space=memory_space))
  #   p = malloc(size)
  #   mem = SE_DeviceMemoryBase(opaque=p, size=size, payload=size)
  #   brk()
  #   return mem
  # bool TpuExecutor_Allocate(SE_StreamExecutor* executor,
  #                           uint64_t size, int64_t memory_space, SE_DeviceMemoryBase* result);
  def Allocate(self: SE_StreamExecutor,
               size: uint64_t, memory_space: int64_t, result: SE_DeviceMemoryBase_p) -> bool:
    print('Allocate', dict(size=size, memory_space=memory_space))
    p = malloc(size)
    if p:
      result.contents.opaque = p
      result.contents.size = size
      result.contents.payload = size
      return True
    else:
      panic('Allocation failed', size, memory_space)
      result.contents.opaque = void_p(None)
      result.contents.size = 0
      result.contents.payload = 0
      return False
  # void TpuExecutor_Deallocate(SE_StreamExecutor* executor,
  #                             SE_DeviceMemoryBase* memory);
  # bool TpuExecutor_GetAllocatorStats(SE_StreamExecutor* executor,
  #                                    SE_AllocatorStats* stats);
  # bool TpuExecutor_DeviceMemoryUsage(SE_StreamExecutor* executor, int64_t* free,
  #                                    int64_t* total);
  #
  # bool TpuExecutor_AllocateStream(SE_StreamExecutor* executor, SE_Stream* stream);
  def AllocateStream(self: SE_StreamExecutor, stream: SE_Stream) -> bool:
    print('AllocateStream', self, stream)
    return True
  # void TpuExecutor_DeallocateStream(SE_StreamExecutor* executor,
  #                                   SE_Stream* stream);
  def DeallocateStream(self: SE_StreamExecutor, stream: SE_Stream):
    print('DeallocateStream', self, stream)
  # bool TpuExecutor_CreateStreamDependency(SE_StreamExecutor* executor,
  #                                         SE_Stream* dependent, SE_Stream* other);
  # void TpuExecutor_GetStatus(SE_StreamExecutor* executor, SE_Stream* stream,
  #                            TF_Status* status);
  #
  # SE_TpuTopology_Core* TpuExecutor_GetCoreLocation(SE_StreamExecutor* executor);
  def GetCoreLocation(self: SE_StreamExecutor) -> SE_TpuTopology_Core:
    return SE_TpuTopology_Core(id=0)
  #
  # void TpuExecutor_AllocateEvent(SE_StreamExecutor* executor, SE_Event* event,
  #                                TF_Status* status);
  # void TpuExecutor_DeallocateEvent(SE_StreamExecutor* executor, SE_Event* event,
  #                                  TF_Status* status);
  # int TpuExecutor_PollForEventStatus(SE_StreamExecutor* executor,
  #                                    SE_Event* event);
  # void TpuExecutor_RecordEvent(SE_StreamExecutor* executor, SE_Stream* stream,
  #                              SE_Event* event, TF_Status* status);
  # void TpuExecutor_WaitForEvent(SE_StreamExecutor* executor, SE_Stream* stream,
  #                               SE_Event* event, TF_Status* status);
  #
  # bool TpuExecutor_AllocateTimer(SE_StreamExecutor* executor, SE_Timer* timer);
  # void TpuExecutor_DeallocateTimer(SE_StreamExecutor* executor, SE_Timer* timer);
  # bool TpuExecutor_StartTimer(SE_StreamExecutor* executor, SE_Stream* stream,
  #                             SE_Timer* timer);
  # bool TpuExecutor_StopTimer(SE_StreamExecutor* executor, SE_Stream* stream,
  #                            SE_Timer* timer);
  #
  # void TpuExecutor_SynchronousMemcpyToHost(SE_StreamExecutor* executor,
  #                                          void* host_dst,
  #                                          const SE_DeviceMemoryBase* device_src,
  #                                          uint64_t size, TF_Status* status);
  # void TpuExecutor_SynchronousMemcpyFromHost(SE_StreamExecutor* executor,
  #                                            SE_DeviceMemoryBase* device_dst,
  #                                            const void* host_src, uint64_t size,
  #                                            TF_Status* status);
  # bool TpuExecutor_MemcpyToHost(SE_StreamExecutor* executor, SE_Stream* stream,
  #                               void* host_dst,
  #                               const SE_DeviceMemoryBase* device_src,
  #                               uint64_t size);
  #
  # bool TpuExecutor_MemcpyFromHost(SE_StreamExecutor* executor, SE_Stream* stream,
  #                                 SE_DeviceMemoryBase* device_dst,
  #                                 const void* host_src, uint64_t size);
  #
  # void TpuExecutor_EnqueueInfeed(SE_StreamExecutor* executor,
  #                                int32_t infeed_queue_index, const uint8_t* data,
  #                                int64_t size, TF_Status* status);
  # void TpuExecutor_DequeueOutfeed(SE_StreamExecutor* executor,
  #                                 int32_t outfeed_queue_index, uint8_t* data,
  #                                 int64_t size, TF_Status* status);
  # void TpuExecutor_WaitForInfeedReady(SE_StreamExecutor* executor,
  #                                     int32_t infeed_queue_index,
  #                                     TF_Status* status);
  # void TpuExecutor_WaitForOutfeedReady(SE_StreamExecutor* executor,
  #                                      int32_t outfeed_queue_index,
  #                                      TF_Status* status);
  #
  # void TpuExecutor_BlockHostUntilDone(SE_StreamExecutor* executor,
  #                                     SE_Stream* stream, TF_Status* status);
  def BlockHostUntilDone(self: SE_StreamExecutor,
                         stream: SE_Stream, status: TF_Status):
    print('BlockHostUntilDone', stream, status)
    status.ok()
  # void TpuExecutor_BlockUntilDoneOrFailed(SE_StreamExecutor* executor,
  #                                         TF_Status* status);
  # void TpuExecutor_SyncAndForgetFailedStreams(SE_StreamExecutor* executor);
  # bool TpuExecutor_SynchronizeAllActivity(SE_StreamExecutor* executor);
  def SynchronizeAllActivity(self: SE_StreamExecutor) -> bool:
    return True
  #
  # void TpuExecutor_UnloadAllPrograms(SE_StreamExecutor* executor,
  #                                    TF_Status* status);
  # void TpuExecutor_EnqueueCompactionOnStreamForHbm(SE_StreamExecutor* executor,
  #                                                  SE_Stream* compaction_stream,
  #                                                  TF_Status* status);

#
# TpuStream / SE_Stream
#
@dataclass
class SE_Stream(TpuType, use_name='TpuStream'):
  parent: SE_StreamExecutor
  # SE_Stream* TpuStream_New(SE_StreamExecutor* parent);
  @classmethod
  def New(cls: SE_Stream, parent: SE_StreamExecutor) -> SE_Stream:
    return cls(parent=parent)
  # void TpuStream_Free(SE_Stream*);
  def Free(self: SE_Stream):
    return delete(self)
  # void* TpuStream_Stream(SE_Stream*);
  # bool TpuStream_Status(SE_Stream*);
  # bool TpuStream_IsSameSharedMemoryLocation(SE_Stream*, SE_Stream*);
  # void TpuStream_EnqueueTransferHostToDevice(SE_Stream* stream,
  #                                            SE_DeviceMemoryBase device_dst,
  #                                            void* host_src, uint64_t size,
  #                                            TF_Status* status);
  # void TpuStream_EnqueueTransferDeviceToHost(SE_Stream* stream,
  #                                            SE_DeviceMemoryBase device_src,
  #                                            void* host_dst, uint64_t size,
  #                                            TF_Status* status);
  # void TpuStream_TpuEnqueueOnDeviceSendRecvLocal(SE_Stream* stream,
  #                                                SE_DeviceMemoryBase send_buffer,
  #                                                SE_DeviceMemoryBase recv_buffer,
  #                                                TF_Status* status);


TPU_C_API_MAX_INLINED = 6
# struct Int64List {
#   union {
#     int64_t* heap;  // owned
#     int64_t inlined[TPU_C_API_MAX_INLINED];
#   };
#   int64_t size;
# };
TPU_C_API_MAX_INLINED_Int64List = int64_t * TPU_C_API_MAX_INLINED

# # @pyembc_union
# @union
# class Int64ListHeap:
#   ptr: ptr_t
#   heap: int64_p
#   inlined: TPU_C_API_MAX_INLINED_Int64List
Int64ListHeap = int64_t * TPU_C_API_MAX_INLINED

class ListBase:
  def view(self, ctype=int32_t):
    if self.size > TPU_C_API_MAX_INLINED:
      p = ctypes.cast(ctypes.addressof(self.u), ctypes.POINTER(ctypes.POINTER(ctype)))[0]
    else:
      p = ctypes.cast(ctypes.addressof(self.u), ctypes.POINTER(ctype))
    return p[0:self.size]


@struct
class Int64List:
  u: Int64ListHeap
  size: int64_t
  # def __len__(self):
  #   return self.size
  # def __getitem__(self, i):
  #   if not isinstance(i, int):
  #     raise TypeError("Indices must be integers")
  #   if not (0 <= i < self.size):
  #     raise IndexError("Index out of range")
  #   if self.size > TPU_C_API_MAX_INLINED:
  #     return self.u.heap[i]
  #   else:
  #     return self.u.inlined[i]

def Int64List_Free(self: Int64List):
  if self.size > TPU_C_API_MAX_INLINED:
    raise NotImplementedError
    # self.u.ptr = free(self.u.ptr)
    # free(ctypes.addressof(self.u))
    # self.u = free(self.u)
  self.size = 0

Int64List.Free = Int64List_Free

def Int64List_Set(self: Int64List, other: Int64List):
  if other.size > TPU_C_API_MAX_INLINED:
    raise NotImplementedError
  for i in range(TPU_C_API_MAX_INLINED):
    self.u[i] = 0
  self.size = 0
  for i in range(other.size):
    self.u[i] = other.u[i]
    self.size += 1
  return self

Int64List.Set = Int64List_Set

# struct BoolList {
#   union {
#     bool* heap;  // owned
#     bool inlined[TPU_C_API_MAX_INLINED];
#   };
#   int64_t size;
# };
# TPU_C_API_MAX_INLINED_BoolList = bool_t * TPU_C_API_MAX_INLINED

# @pyembc_union
# class BoolListHeap:
#   heap: int64_p
#   inlined: TPU_C_API_MAX_INLINED_BoolList
BoolListHeap = bool_t * TPU_C_API_MAX_INLINED

@struct
class BoolList:
  u: BoolListHeap
  size: int64_t

def BoolList_Free(self: BoolList):
  if self.size > TPU_C_API_MAX_INLINED:
    raise NotImplementedError
    # self.u.ptr = free(self.u.ptr)
    # free(ctypes.addressof(self.u))
    # self.u = free(self.u)
  self.size = 0

BoolList.Free = BoolList_Free

def BoolList_Set(self: BoolList, other: BoolList):
  if other.size > TPU_C_API_MAX_INLINED:
    raise NotImplementedError
  for i in range(TPU_C_API_MAX_INLINED):
    self.u[i] = False
  self.size = 0
  for i in range(other.size):
    self.u[i] = other.u[i]
    self.size += 1
  return self

BoolList.Set = BoolList_Set

# typedef struct XLA_Tile {
#   Int64List dimensions;
# } XLA_Tile;
@struct
class XLA_Tile:
  dimensions: Int64List

def XLA_Tile_Free(self: XLA_Tile):
  self.dimensions.Free()

XLA_Tile.Free = XLA_Tile_Free

def XLA_Tile_Set(self: XLA_Tile, other: XLA_Tile):
  self.dimensions.Set(other.dimensions)
  return self

XLA_Tile.Set = XLA_Tile_Set

# struct TileList {
#   union {
#     XLA_Tile* heap;  // owned
#     XLA_Tile inlined[TPU_C_API_MAX_INLINED];
#   };
#   int64_t size;
# };
# TPU_C_API_MAX_INLINED_TileList = XLA_Tile * TPU_C_API_MAX_INLINED

# @pyembc_union
# class TileListHeap:
#   heap: int64_p
#   inlined: TPU_C_API_MAX_INLINED_TileList
TileListHeap = XLA_Tile * TPU_C_API_MAX_INLINED

@struct
class TileList:
  u: TileListHeap
  size: int64_t

def TileList_Free(self: TileList):
  for i in range(self.size):
    self.u[i].Free()
  if self.size > TPU_C_API_MAX_INLINED:
    raise NotImplementedError
  #   u, self.u = self.u, None
  #   free(ctypes.addressof(u))
  self.size = 0

TileList.Free = TileList_Free

def TileList_Set(self: TileList, other: TileList):
  if other.size > TPU_C_API_MAX_INLINED:
    raise NotImplementedError
  for i in range(TPU_C_API_MAX_INLINED):
    self.u[i].Set(XLA_Tile())
  self.size = 0
  for i in range(other.size):
    self.u[i].Set(other.u[i])
    self.size += 1
  return self

TileList.Set = TileList_Set

# typedef struct XLA_Layout {
#   int format;
#   Int64List minor_to_major;
#   TileList tiles;
#   int64_t element_size_in_bits;
#   int64_t memory_space;
# } XLA_Layout;

@struct
class XLA_Layout:
  format: int_t
  minor_to_major: Int64List
  tiles: TileList
  element_size_in_bits: int64_t
  memory_space: int64_t

def XLA_Layout_Set(self: XLA_Layout, other: XLA_Layout):
  self.format = other.format
  self.minor_to_major.Set(other.minor_to_major)
  self.tiles.Set(other.tiles)
  self.element_size_in_bits = other.element_size_in_bits
  self.memory_space = other.memory_space
  return self

XLA_Layout.Set = XLA_Layout_Set

# void Free(XLA_Layout* c_layout) {
def XLA_Layout_Free(self: XLA_Layout):
  # if (c_layout->minor_to_major.size > TPU_C_API_MAX_INLINED) {
  #   delete[] c_layout->minor_to_major.heap;
  # }
  self.minor_to_major.Free()
  # if (c_layout->tiles.size > TPU_C_API_MAX_INLINED) {
  #   delete[] c_layout->tiles.heap;
  # }
  self.tiles.Free()
# }

XLA_Layout.Free = XLA_Layout_Free

# // Represents an XLA shape tree.
# typedef struct XLA_Shape {
#   int element_type;
#   Int64List dimensions;
#   BoolList dynamic_dimensions;
#   XLA_Shape* tuple_shapes;  // owned
#   int ntuple_shapes;
#   XLA_Layout layout;
# } XLA_Shape;

#XLA_Shape_p = ctypes.POINTER(XLA_Shape)

@struct
class XLA_Shape:
  element_type: int_t
  dimensions: Int64List
  dynamic_dimensions: BoolList
  tuple_shapes: XLA_Shape_p
  ntuple_shapes: int_t
  layout: XLA_Layout

def XLA_Shape_Set(self: XLA_Shape, other: XLA_Shape):
  # brk()
  # self.Free() # TODO: Figure out how not to leak
  self.element_type = other.element_type
  self.dimensions.Set(other.dimensions)
  self.dynamic_dimensions.Set(other.dynamic_dimensions)
  self.ntuple_shapes = other.ntuple_shapes
  if self.ntuple_shapes > 0:
    tuple_shapes_p = malloc(ctypes.sizeof(XLA_Shape) * self.ntuple_shapes)
    self.tuple_shapes = ctypes.cast(tuple_shapes_p, XLA_Shape_p)
    for i in range(self.ntuple_shapes):
      self.tuple_shapes[i].Set(other.tuple_shapes[i])
  self.layout.Set(other.layout)
  return self

XLA_Shape.Set = XLA_Shape_Set

# void Free(XLA_Shape* c_shape) {
def XLA_Shape_Free(self: XLA_Shape):
  #   if (c_shape->dimensions.size > TPU_C_API_MAX_INLINED) {
  #     delete[] c_shape->dimensions.heap;
  #   }
  self.dimensions.Free()
  #   if (c_shape->dynamic_dimensions.size > TPU_C_API_MAX_INLINED) {
  #     delete[] c_shape->dynamic_dimensions.heap;
  #   }
  self.dynamic_dimensions.Free()
  # if (c_shape->ntuple_shapes > 0) {
  if self.ntuple_shapes > 0:
    # for (int i = 0; i < c_shape->ntuple_shapes; ++i) {
    #   Free(&c_shape->tuple_shapes[i]);
    # }
    n, self.ntuple_shapes = self.ntuple_shapes, 0
    for i in range(n):
      self.tuple_shapes[i].Free()
    # delete[] c_shape->tuple_shapes;
    # free(self.tuple_shapes) # TODO: leak?
  # }
  # if (c_shape->layout.format != xla::INVALID_FORMAT) {
  #   Free(&c_shape->layout);
  # }
  if self.layout.format != 0: # xla.INVALID_FORMAT
    self.layout.Free()
  # }

XLA_Shape.Free = XLA_Shape_Free


XLA_Shape_p = ctypes.POINTER(XLA_Shape)

# h0 = Int64ListHeap(int64_p(ctypes.c_long(0)), TPU_C_API_MAX_INLINED_Int64List(*(ctypes.c_long(i) for i in (0, 0, 0, 0, 0, 0))))
# h1 = Int64List(h0, int64_t(0))
# b0 = BoolListHeap(int64_p(ctypes.c_long(0)), TPU_C_API_MAX_INLINED_BoolList(*(ctypes.c_bool(i) for i in (0, 0, 0, 0, 0, 0))))
# b1 = BoolList(b0, int64_t(0))
# brk()

#     _descriptor.EnumValueDescriptor(
#       name='PRIMITIVE_TYPE_INVALID', index=0, number=0,
#       name='PRED', index=1, number=1,
#       name='S8', index=2, number=2,
#       name='S16', index=3, number=3,
#       name='S32', index=4, number=4,
#       name='S64', index=5, number=5,
#       name='U8', index=6, number=6,
#       name='U16', index=7, number=7,
#       name='U32', index=8, number=8,
#       name='U64', index=9, number=9,
#       name='F16', index=10, number=10,
#       name='F32', index=11, number=11,
#       name='BF16', index=12, number=16,
#       name='F64', index=13, number=12,
#       name='C64', index=14, number=15,
#       name='C128', index=15, number=18,
#       name='TUPLE', index=16, number=13,
#       name='OPAQUE_TYPE', index=17, number=14,
#       name='TOKEN', index=18, number=17,
XLA_name2type = dict(
  PRIMITIVE_TYPE_INVALID=0,
  PRED=1,
  S8=2,
  S16=3,
  S32=4,
  S64=5,
  U8=6,
  U16=7,
  U32=8,
  U64=9,
  F16=10,
  F32=11,
  BF16=16,
  F64=12,
  C64=15,
  C128=18,
  TUPLE=13,
  OPAQUE_TYPE=14,
  TOKEN=17)
XLA_type2name = {v: k for k, v in XLA_name2type.items()}
XLA_name2size = dict(
  PRIMITIVE_TYPE_INVALID=-1,
  PRED=-1,
  S8=1,
  S16=2,
  S32=4,
  S64=8,
  U8=1,
  U16=2,
  U32=4,
  U64=8,
  F16=2,
  F32=4,
  BF16=2,
  F64=8,
  C64=8,
  C128=16,
  TUPLE=-1,
  OPAQUE_TYPE=-1,
  TOKEN=-1)

#
# TpuTransferManager / XLA_TransferManager
#
@dataclass
class XLA_TransferManager(NewFree, TpuType, use_name="TpuTransferManager"):
  # XLA_TransferManager* TpuTransferManager_New();
  @classmethod
  def New(cls) -> XLA_TransferManager:
    return XLA_TransferManager()
  # void TpuTransferManager_Free(XLA_TransferManager* manager);
  def Free(self: XLA_TransferManager):
    return delete(self)
  # SE_PlatformId TpuTransferManager_PlatformId(XLA_TransferManager* manager);
  # void TpuTransferManager_HostShapeToDeviceShape(XLA_TransferManager* manager,
  #                                                XLA_Shape* host_shape,
  #                                                XLA_Shape* device_shape);
  def HostShapeToDeviceShape(self: XLA_TransferManager,
                             host_shape: XLA_Shape,
                             device_shape: XLA_Shape):
    device_shape.Set(host_shape)
  # void TpuTransferManager_TransferLiteralToDeviceAsync(
  #     XLA_TransferManager* manager, SE_Stream* stream, XLA_Literal* literal,
  #     XLA_ShapedBuffer* device_buffer, TF_Status* status);
  # void TpuTransferManager_TransferLiteralFromDevice(
  #     XLA_TransferManager* manager, SE_Stream* stream,
  #     XLA_ShapedBuffer* device_buffer, XLA_Literal* literal,
  #     XLA_StatusCallbackFn callback, void* ctx);
  # int64_t TpuTransferManager_GetByteSizeRequirement(XLA_TransferManager* manager,
  #                                                   XLA_Shape* shape);
  def GetByteSizeRequirement(self: XLA_TransferManager,
                             shape: XLA_Shape):
    type = XLA_type2name[shape.element_type]
    total = XLA_name2size[type]
    if total < 0:
      panic('Shape type not implemented', shape)
    for i in range(shape.dimensions.size):
      dim = shape.dimensions.u[i]
      total *= dim
    return total

  # void TpuTransferManager_ChooseCompactLayoutForShape(
  #     XLA_TransferManager* manager, XLA_Shape* host_shape, XLA_Shape* output,
  #     TF_Status* status);
  def ChooseCompactLayoutForShape(
          self: XLA_TransferManager, host_shape: XLA_Shape, output: XLA_Shape,
          status: TF_Status):
    output.Set(host_shape)
    status.ok()
  # bool TpuTransferManager_CanShapedBufferBeAccessedNow(
  #     XLA_TransferManager* manager, SE_StreamExecutor* executor,
  #     XLA_ShapedBuffer* device_buffer);
  # bool TpuTransferManager_CanBufferBeAccessedNow(
  #     XLA_TransferManager* manager, SE_StreamExecutor* executor,
  #     SE_DeviceMemoryBase* device_buffer);
  # void TpuTransferManager_WriteSingleTupleIndexTable(
  #     XLA_TransferManager* manager, SE_Stream* stream,
  #     SE_DeviceMemoryBase* elements, size_t elements_len, XLA_Shape* shape,
  #     SE_DeviceMemoryBase* region, TF_Status* status);
  # void TpuTransferManager_GetInfeedLayout(XLA_Shape* shape,
  #                                         XLA_Shape* infeed_shape);
  # void TpuTransferManager_LinearizeToBuffers(
  #     XLA_TransferManager* manager, XLA_Literal* c_literal, char*** buffers_array,
  #     int64_t** buffers_size, int64_t* buffers_array_size, TF_Status* status);
  # void TpuTransferManager_FreeBuffers(char** buffers_array, int64_t* buffers_size,
  #                                     int64_t buffers_array_size);
  # void TpuTransferManager_TransferLiteralToInfeed(XLA_TransferManager* manager,
  #                                                 SE_StreamExecutor* executor,
  #                                                 XLA_Literal* c_literal,
  #                                                 TF_Status* status);
  # void TpuTransferManager_TransferBuffersToInfeed(XLA_TransferManager* manager,
  #                                                 SE_StreamExecutor* executor,
  #                                                 uint32_t** buffers_array,
  #                                                 int64_t* buffers_size_in_uint32,
  #                                                 int64_t buffers_array_size,
  #                                                 TF_Status* status);
  # void TpuTransferManager_TransferLiteralFromOutfeed(
  #     XLA_TransferManager* manager, SE_StreamExecutor* executor,
  #     XLA_Shape* shape /*deprecated*/, XLA_Literal* c_literal, TF_Status* status);
  # void TpuTransferManager_ResetDevices(XLA_TransferManager* manager,
  #                                      SE_StreamExecutor** executors,
  #                                      int64_t num_executors, TF_Status* status);
  # void TpuTransferManager_ReadDynamicShapes(SE_Stream* stream,
  #                                           XLA_ShapedBuffer* buffer,
  #                                           const XLA_Shape& original_shape,
  #                                           XLA_Shape* updated_shape,
  #                                           TF_Status* status);

#
# TpuComputationPlacer / XLA_ComputationPlacer
#
@dataclass
class XLA_ComputationPlacer:
  pass

class TpuComputationPlacer(TpuType, wraps=XLA_ComputationPlacer):

  # XLA_ComputationPlacer* TpuComputationPlacer_New();
  @classmethod
  def New(cls) -> XLA_ComputationPlacer:
    return XLA_ComputationPlacer()

  # void TpuComputationPlacer_Free(XLA_ComputationPlacer* placer);
  def Free(self: XLA_ComputationPlacer):
    return delete(self)

  # // `assignment` should be a preallocated array of size `replicate_count` *
  # // `computation_count`. The assignment will be constructed as a 2D array where
  # // assignment[replica][computation] = device_id.
  # void TpuComputationPlacer_AssignDevices(XLA_ComputationPlacer* placer,
  #                                         int replica_count,
  #                                         int computation_count, int* assignment,
  #                                         TF_Status* status);
  def AssignDevices(self: XLA_ComputationPlacer,
                    replica_count: int,
                    computation_count: int,
                    assignment: int_out,
                    status: TF_Status):
    print('AssignDevices', replica_count, computation_count)
    i = 0
    for replica in range(replica_count):
      for computation in range(computation_count):
        # v = assignment[replica*replica_count + computation]
        # print(f'{replica},{computation} = {v}')
        assignment[replica*replica_count + computation] = i
        print(f'{replica},{computation} = {i}')
        i += 1
        i %= len(SE_Platform.get().devices)
    status.ok()
  # void TpuComputationPlacer_AssignLocalDevices(SE_TpuTopology_Host* host,
  #                                              int replica_count,
  #                                              int computation_count,
  #                                              int* assignment,
  #                                              TF_Status* status);
  def AssignLocalDevices(self: SE_TpuTopology_Host,
                         replica_count: int,
                         computation_count: int,
                         assignment: int_out,
                         status: TF_Status):
    print('AssignLocalDevices', replica_count, computation_count)
    i = 0
    for replica in range(replica_count):
      for computation in range(computation_count):
        # v = assignment[replica*replica_count + computation]
        # print(f'{replica},{computation} = {v}')
        assignment[replica*replica_count + computation] = i
        print(f'{replica},{computation} = {i}')
        i += 1
        i %= len(SE_Platform.get().devices)
    status.ok()



@libtpujesus.set_callback
def api_callback(name, stack_pointer, arg0, *args):
  result = 0
  try:
    global stack_ptr
    global _argv
    _argv = args
    ptr = ctypes.cast(stack_pointer, stackptr_t)
    stack_ptr = ptr
    f = globals().get(name, None)
    if not f:
      #print(f"\nTPU API function not implemented: {name}{args!r}\n")
      panic("TPU API function not implemented", name)
    else:
      sig = inspect.signature(f)
      args = _argv[:len(sig.parameters)]
      # print(f"TPU API function implemented! {name}{args!r}")
      vals = tuple([argv(f, i) for i in range(len(sig.parameters))])
      print(f"\nCALL: {name}")
      for i, val in enumerate(vals):
        print(f"\targ[{i}]={val!r}")
      result = f(*vals)
      print(f"-> {result!r}\n")
      # print(f"TPU API function returning {result}: {name}{vals!r}")
  except:
    traceback.print_exc()
    pdb.post_mortem(sys.exc_info()[2])
    panic("Unhandled error")
  if result is None:
    result = 0
  elif isinstance(result, (bool, int)):
    result = int(result)
  elif isinstance(result, list) and len(result) == 1 and is_cstruct(result[0]):
    # result = ctypes.pointer(result[0])
    # pin(result)
    # result = ctypes.addressof(result)
    # result = 0
    result = pin(result[0])
    result = 0
    raise NotImplementedError
  elif is_cstruct(result):
    pin(result) # TODO: is this necessary?
    result = ctypes.addressof(result)
  elif isinstance(result, (Wrappable, tuple)) or is_dataclass_instance(result):
    result = new(result)
  elif is_cdata(result):
    result = cvalue(result)
  else:
    panic("Don't know how to return", result)
  print(f"finally returning {result}: {name}{vals!r}")
  return result,

def configure_library_path():
  print('libtpu.configure_library_path()')

sys.stdout = __saved_stdout__
