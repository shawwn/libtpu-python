import libtpujesus
import os
os.environ['TPU_LIBRARY_PATH'] = libtpujesus.__file__
os.environ['CLOUD_TPU_TASK_ID'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import jax
import jax.numpy as jnp
devices = jax.devices()
print(devices)
dev = devices[0]
print(dir(dev))

from jax._src.lib import xla_bridge as xb

be = xb.get_backend()
print(dir(be))
ass = be.get_default_device_assignment(1)
print(ass)
ass2 = be.get_default_device_assignment(2, 2)
print(ass2)

buf = jax.device_put(1)

breakpoint()
qq = jnp.zeros((1,2))
print('ok')

#print(jnp.zeros((1,2)))
