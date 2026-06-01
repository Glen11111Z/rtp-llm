load("@arch_config//:arch_select.bzl", "copy_all_so")
load("//bazel:defs.bzl", "copy_files")

package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

# Copy flashinfer sub shared libs (the ones librtp_compute_ops.so links
# against via -lflashinfer_single_prefill / _single_decode / _batch_*) into
# bazel-bin/rtp_llm/libs/ so they end up in the wheel next to
# librtp_compute_ops.so. The .so's RPATH already contains $ORIGIN, so this
# is sufficient for runtime resolution after pip install.
#
# Use select to pick from @flashinfer_ppu on PPU or @flashinfer_cpp otherwise,
# without requiring aliases defined in arch_select.bzl.
_FLASHINFER_SO_NAMES = [
    "flashinfer_single_prefill",
    "flashinfer_single_decode",
    "flashinfer_batch_paged_prefill",
    "flashinfer_batch_paged_decode",
    "flashinfer_batch_ragged_prefill",
]

[
    genrule(
        name = "lib" + _so + "_so",
        srcs = select({
            "@rtp_llm//internal_source/RTP_LLM-PPU:use_ppu": ["@flashinfer_ppu//:" + _so],
            "//conditions:default": ["@flashinfer_cpp//:" + _so],
        }),
        outs = ["lib" + _so + ".so"],
        cmd = "cp $(SRCS) $(@D)",
        tags = ["no-remote"],
    )
    for _so in _FLASHINFER_SO_NAMES
]

_FLASHINFER_WHL_LIBS = [":lib" + _so + "_so" for _so in _FLASHINFER_SO_NAMES]

filegroup(
    name = "flashinfer_whl_libs",
    srcs = [],
    data = select({
        "@//:using_cuda12_9_x86": _FLASHINFER_WHL_LIBS,
        "@//:using_cuda12_arm": _FLASHINFER_WHL_LIBS,
        "@//:cuda_pre_12_9": _FLASHINFER_WHL_LIBS,
        "@rtp_llm//internal_source/RTP_LLM-PPU:use_ppu": _FLASHINFER_WHL_LIBS,
        "//conditions:default": [],
    }),
)

# copy 3fs libs
copy_files(
    name = "copy_hf3fs_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/3fs:hf3fs_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_remote_kv_cache_manager_client_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_acclep_libs",
    srcs = select({
        "//rtp_llm/models_py/bindings/cuda/ops:use_accl_ep": ["//3rdparty/accl_ep:accl_ep_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "frontend_libs",
    srcs = [],
    data = [
        ":libth_transformer_config_so",
    ],
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":copy_acclep_libs",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
    ],
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":librtp_compute_ops_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
        ":flashinfer_whl_libs",
    ],
)
