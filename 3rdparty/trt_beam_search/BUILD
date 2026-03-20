load("//:def.bzl", "rpm_library", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trt_beam_search_impl",
    srcs = glob([
        "*.cu",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        ":trt_beam_search_header",
        "//rtp_llm/cpp/cuda:trt_utils",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/utils:math_utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)

cc_library(
    name = "trt_beam_search_rocm_impl",
    srcs = glob([
        "*.cu",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//rtp_llm/cpp/rocm:rocm_types_hdr",
        "//rtp_llm/cpp/rocm:rocm_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
    ],
    copts = rocm_copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)

cc_library(
    name = "trt_beam_search_header",
    hdrs = glob([
        "*.h",
        "**/*.h",
        "**/*.cuh",
    ]),
    deps = select({
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
    copts = any_cuda_copts(),
    visibility = ["//visibility:public"],
)
