// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>
#include <random>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_elementwise.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F16;
using BDataType = F16;

using UnaryScale  = ck::tensor_operation::element_wise::Scale;
using UnarySquare = ck::tensor_operation::element_wise::UnarySquare;
using UnaryScaleSquare =
    ck::tensor_operation::element_wise::UnaryCombinedOp<UnarySquare, UnaryScale>;
using DeviceElementwisePermuteInstance = ck::tensor_operation::device::DeviceElementwiseImpl<
    ck::Tuple<ADataType>, // InDataTypeTuple
    ck::Tuple<BDataType>, // OutDataTypeTuple
    UnaryScaleSquare,     // UnaryScaleSquare
    4,                    // NumDim
    256,                  // BlockSize
    128,                  // M0PerBlock
    128,                  // M1PerBlock
    8,                    // M0PerThread
    8,                    // M1PerThread
    ck::Sequence<1, 0>,   // ThreadClusterArrangeOrder
    ck::Sequence<8>,      // InScalarPerVectorSeq
    ck::Sequence<8>>;     // OutScalarPerVectorSeq

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    std::vector<std::size_t> nchw = {16, 8, 32, 64};
    std::vector<std::size_t> nhwc = {16, 32, 64, 8};
    std::array<ck::index_t, 4> ab_lengths;
    std::array<ck::index_t, 4> a_strides = {1,
                                            static_cast<int>(nchw[0]),
                                            static_cast<int>(nchw[0] * nchw[1]),
                                            static_cast<int>(nchw[0] * nchw[1] * nchw[2])};

    std::array<ck::index_t, 4> b_strides = {1,
                                            static_cast<int>(nhwc[0] * nhwc[1] * nhwc[2]),
                                            static_cast<int>(nhwc[0]),
                                            static_cast<int>(nhwc[0] * nhwc[1])};
    ck::ranges::copy(nchw, ab_lengths.begin());

    std::array<Tensor<ADataType>, 1> as = {Tensor<ADataType>(ab_lengths, a_strides)};
    Tensor<ADataType>& a                = as[0];
    Tensor<BDataType> b(ab_lengths, b_strides);
    float scale = 1.f;
    auto i      = 0;
    std::mt19937 gen(11939);
    std::uniform_int_distribution<int> dis(0, 1);
    for(std::size_t w = 0; w < a.mDesc.GetLengths()[3]; ++w)
        for(std::size_t h = 0; h < a.mDesc.GetLengths()[2]; ++h)
            for(std::size_t c = 0; c < a.mDesc.GetLengths()[1]; ++c)
                for(std::size_t n = 0; n < a.mDesc.GetLengths()[0]; ++n)
                {
                    a.mData[(n * nchw[1] * nchw[2] * nchw[3]) + (c * nchw[2] * nchw[3]) +
                            (h * nchw[3]) + w] = i;
                    i                          = dis(gen);
                }

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument =
        broadcastPermute.MakeArgumentPointer(ab_lengths,
                                             {a_strides},
                                             {b_strides},
                                             input,
                                             output,
                                             UnaryScaleSquare{UnarySquare{}, UnaryScale{scale}});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A (nchw): " << a.mDesc << std::endl;
    std::cout << "B (nhwc): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(5) * nchw[0] * nchw[1] * nchw[2] * nchw[3];

    std::size_t num_btype =
        (2 * sizeof(ADataType) + sizeof(BDataType)) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]);
    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        Tensor<BDataType> host_b(ab_lengths, b_strides);
        using ReferenceElementwiseInstance = ck::tensor_operation::host::
            ReferenceElementwise<1, ADataType, BDataType, UnaryScaleSquare>;
        auto ref_elementwise = ReferenceElementwiseInstance{};
        auto ref_invoker     = ref_elementwise.MakeInvoker();

        auto ref_argument = ref_elementwise.MakeArgument(
            as, host_b, UnaryScaleSquare{UnarySquare{}, UnaryScale{scale}});
        ref_invoker.Run(ref_argument);

        b_device_buf.FromDevice(b.mData.data());
        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
