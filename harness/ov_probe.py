"""Probe: what precision/ISA OpenVINO CPU resolves to. Run locked with venv python."""
import json, os, sys
import openvino as ov
import openvino.properties as p
import openvino.properties.hint as h
core = ov.Core()
props = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
m = core.read_model("/home/user/Infer-opts/models/ov/Qwen3.5-2B-int4-ov/openvino_language_model.xml")
cm = core.compile_model(m, "CPU", props)
for k in ["INFERENCE_PRECISION_HINT", "EXECUTION_MODE_HINT", "PERFORMANCE_HINT", "NUM_STREAMS", "INFERENCE_NUM_THREADS",
          "KV_CACHE_PRECISION", "DYNAMIC_QUANTIZATION_GROUP_SIZE", "ENABLE_HYPER_THREADING", "AFFINITY"]:
    try: print(k, "=", cm.get_property(k))
    except Exception as e: print(k, "ERR", str(e)[:80])
print("FULL_DEVICE_NAME", core.get_property("CPU", "FULL_DEVICE_NAME"))
print("OPTIMIZATION_CAPABILITIES", core.get_property("CPU", "OPTIMIZATION_CAPABILITIES"))
