import re
import json
import os
import json
import glob

for object_path in glob.glob("target/debug/build/backend-uzu-*/out/metal/*/*.objectinfo"):
  with open(object_path) as fd:
    object: dict = json.loads(fd.read())

  if not object["kernels"]:
    continue

  generated_imports = set()

  def mtl2slang(mtl: str):
    if mtl.startswith("uzu::"):
      mtls = mtl.split("::")
      assert len(mtls) == 3
      generated_imports.add(f"import generated.{mtls[1]};")
      return mtls[2]
    else:
      return {"float": "float", "float4": "float4", "half": "half", "bfloat": None, "bfloat16_t": None, "bool": "bool", "uint8_t": "uint8_t", "uint": "uint", "_atomic<uint>": "uint", "int32_t": "int", "int": "int", "uint32_t": "uint", "uint64_t": "uint64_t", "ArgmaxPair": "ArgmaxPair"}[mtl]

  def rust2slang(rust: str):
    return {"u32": "uint"}[rust]

  kernels = []
  emitted_constants = False
  for kernel_json in object["kernels"]:
    if not kernel_json["public"]:
      continue

    kernel_name: str = kernel_json["name"]
    function_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', kernel_name)).lower()
    kernel = []

    if not emitted_constants:
      emitted_constants = True
      match object["src_rel_path"]:
        case "attention/attention_gemm.metal": kernel.extend(["static const uint BLOCK_QUERY_ROWS = 32;", ""])
        case "attention/attention_single_pass.metal":
          kernel.extend([
            "static const uint SEQUENCE_BLOCK_SIZE = 32;",
            "static const uint HEAD_BLOCK_SIZE = 32;",
            "",
          ])
        case "audio/norm_ncs.metal": kernel.extend(["static const uint AUDIO_NORM_NCS_MAX_SIMDS = 32;", ""])
        case "delta_net/update.metal": kernel.extend(["static const uint METAL_SIMD_SIZE = 32;", ""])
        case "layer_norm/layer_norm.metal": kernel.extend(["static const uint METAL_SIMD_SIZE = 32;", ""])
        case "moe/counts_offsets_fused.metal":
          kernel.extend([
            "static const uint BLOCK_SIZE = 128;",
            "static const uint TILE_E = 512;",
            "",
          ])
        case "moe/experts_two_pass_prefill.metal":
          kernel.extend([
            "static const uint PASSA_BM = 16;",
            "static const uint PASSA_BN = 32;",
            "static const uint PASSA_BK = 64;",
            "static const uint PASSA_TG_PAD = 4;",
            "static const uint PASSB_BM = 16;",
            "static const uint PASSB_BN = 64;",
            "static const uint PASSB_BK = 64;",
            "static const uint PASSB_TG_PAD = 4;",
            "",
          ])
        case "moe/router_topk.metal":
          kernel.extend([
            "static const uint THREADS_PER_TG = 256;",
            "static const uint MAX_EXPERTS = 512;",
            "",
          ])
        case "moe/scatter_buckets.metal":
          kernel.extend([
            "static const uint BLOCK_SIZE = 256;",
            "static const uint TILE_E = 512;",
            "static const uint SIMD_WIDTH = 32;",
            "static const uint NUM_SG = BLOCK_SIZE / SIMD_WIDTH;",
            "",
          ])
        case "quant_matmul/qmv_fast.metal": kernel.extend(["static const uint METAL_SIMD_SIZE = 32;", ""])
        case "rms_norm/rms_norm.metal": kernel.extend(["static const uint METAL_SIMD_SIZE = 32;", ""])
        case "sampling/argmax.metal": kernel.extend(["static const uint BLOCK_SIZE = 1024;", ""])
        case "sampling/min_p.metal" | "sampling/top_k.metal" | "sampling/top_p.metal":
          kernel.extend([
            "static const uint BLOCK_SIZE = 1024;",
            "static const uint BLOCK_SIZE_IN_SIMDS = BLOCK_SIZE / 32;",
            "",
          ])
        case _: pass

    variants = []
    variant_keys = set()
    for variant in kernel_json["variants"] or []:
      variant_keys.add(variant["name"])
      variants.append(f"{variant["name"]}: __BuiltinFloatingPointType" if variant["ty"] == "Type" else f"let {variant["name"]}: {rust2slang(variant["ty"]["Value"])}")
      kernel.append(f"[[Variants(\"{variant["name"]}\", \"{', '.join(filter(lambda x: x is not None, map(mtl2slang if variant["ty"] == "Type" else (lambda x: x), variant["variants"])))}\")]]")

    for argument in kernel_json["arguments"]:
      metal_type: str = argument["c_type"].split()
      if "threadgroup" not in metal_type:
        continue
      assert metal_type[0] == "threadgroup", metal_type[0]
      assert metal_type[2] in ["*", "&"], metal_type[2]

      metal_annotation = argument["annotation"]
      assert metal_annotation is None, metal_annotation

      metal_source: str = argument["source"]

      if metal_type[2] == "*":
        tgs = metal_source[metal_source.index('[')+1:metal_source.index(']')]
      else:
        tgs = "1"

      shared_type = mtl2slang(metal_type[1]) if metal_type[1] not in variant_keys else metal_type[1]
      variants.append(f"{argument["name"]}: IGroupShared<{shared_type}>")
      kernel.append(f"[[GroupShared({json.dumps(argument["name"])}, {json.dumps(tgs)})]]")

    for constraint in kernel_json["constraints"]:
      kernel.append(f"[[Constraint({json.dumps(constraint)})]]")

    kernel.append(f"[[Public]] [[Kernel]] func {kernel_name}<{', '.join(variants)}>(" if variants else f"[[Public]] [[Kernel]] func {kernel_name}(")

    for argument in kernel_json["arguments"]:
      metal_type: str = argument["c_type"].split()
      metal_annotation = argument["annotation"]
      metal_source: str = argument["source"]

      optional = None
      specialize = False
      if metal_annotation:
        ann_key = metal_annotation[0]

        if ann_key == "dsl.groups":
          kernel.append(f"  [[Groups(\"{metal_annotation[1]}\")]] uint {argument['name']},")
          continue
        if ann_key == "dsl.threads":
          kernel.append(f"  [[Threads(\"{metal_annotation[1]}\")]] uint {argument['name']},")
          continue
        if ann_key == "dsl.axis":
          kernel.append(f"  [[Axis(\"{metal_annotation[1]}\", \"{metal_annotation[2]}\")]] uint {argument['name']},")
          continue
        elif ann_key == "dsl.optional":
          assert len(metal_annotation) == 2
          optional = metal_annotation[1]
        elif ann_key == "dsl.specialize":
          specialize = True
        else:
          raise NotImplementedError(metal_annotation)

      match metal_type:
        case ["const", "device", sdt, "*"]:
          slang_type = f"StructuredBuffer<{mtl2slang(sdt) if sdt not in variant_keys else sdt}>"
        case ["device", sdt, "*"]:
          slang_type = f"RWStructuredBuffer<{mtl2slang(sdt) if sdt not in variant_keys else sdt}>"
        case ["const", "constant", sdt, "*"]:
          try:
            slang_sz = metal_source[metal_source.index('[')+1:metal_source.index(']')]
            slang_type = f"{mtl2slang(sdt)}[{slang_sz}]"
          except:
            slang_type = f"[[Constant]] StructuredBuffer<{mtl2slang(sdt)}>"
        case ["const", "constant", sdt, "&"]:
          slang_type = mtl2slang(sdt)
        case ["const", sdt] if specialize:
          slang_type = mtl2slang(sdt)
        case ["const", "unsigned", "int"] if specialize:
          slang_type = "uint"
        case _ if "threadgroup" in metal_type:
          continue
        case ["const", "ThreadContext"]:
          slang_type = "ThreadContext"
        case _:
          raise NotImplementedError(metal_type)

      kernel.append(f"  {f'[[Optional("{optional}")]] ' if optional is not None else ''}{'[[Specialize]] ' if specialize else ''}{slang_type} {argument["name"]},")

    kernel.append(") {")
    kernel.append("  // TODO")
    kernel.append("}")

    kernels.append("\n".join(kernel))

  gen_contents = [
    "import definitions;",
    *sorted(generated_imports),
    "",
    "\n\n".join(kernels),
  ]

  src_rel_path: str = object["src_rel_path"]
  assert src_rel_path.endswith(".metal")
  src_rel_components = src_rel_path[:-len(".metal")].split("/")

  os.makedirs("crates/backend-uzu/src/backends/slang/" + "/".join(src_rel_components[:-1]), exist_ok=True)

  with open("crates/backend-uzu/src/backends/slang/" + "/".join(src_rel_components) + ".slang", "w") as fd:
    fd.write('\n'.join(filter(lambda x: x is not None, gen_contents)))
