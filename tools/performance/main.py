import random
from tinygrad.runtime.ops_metal import MetalDevice, from_ns_str, MetalBuffer
from tinygrad import Device

random.seed(1337)

device = Device.default
assert isinstance(device, MetalDevice)

print(f"Device: {from_ns_str(device.sysdevice.name())}")
print("---")

def alloc(size: int) -> MetalBuffer:
  return device.allocator.alloc(size)

def run_and_time(src: str, name: str, *bufs: MetalBuffer, global_size: int, local_size: int) -> float:
  lib = device.compiler.compile_cached(src)
  prg = device.runtime(name, lib)
  # device.compiler.disassemble(lib)
  return prg(*bufs, global_size=(global_size, 1, 1), local_size=(local_size, 1, 1), wait=True)

P = 4
N = 1024

G = 1024 * 1024
L = 1024

t = run_and_time(f'''
using namespace metal;

[[max_total_threads_per_threadgroup({L})]] kernel void flops(device half* buf, uint pos [[thread_position_in_grid]]) {{
  {"\n".join([f"half x{i} = buf[pos * {P} + {i}];" for i in range(P)])}
  {"\n".join([f"x{i} = metal::precise::fma(x{i}, {random.random()}f, {random.random()}f);" for i in range(P)] * N)}
  {"\n".join([f"buf[pos * {P} + {i}] = x{i};" for i in range(P)])}
}}
''', "flops", alloc(2 * P * G * L), global_size=G, local_size=L)
print(f"{((N * P * 2 * G * L) / 1e9)/t:.1f} GFLOP/S")
