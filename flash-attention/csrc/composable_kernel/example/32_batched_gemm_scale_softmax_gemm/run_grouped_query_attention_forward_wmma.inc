// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 64;
    ck::index_t O = 64;

    // Output shape C[G0, M, G1, O]. Batch dim, outer dim, inner dim must match GEMM shape
    // C_g0_g1_m_o = reshape(C_g_m_o, [g0, g1, m, o])
    // C_g0_m_g1_o = permute(C_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t G0      = 4;
    ck::index_t G1      = 16;
    ck::index_t KV_head = QueryGroupNumber;

    float alpha = 1;

    bool input_permute  = false;
    bool output_permute = true;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        alpha = std::stof(argv[10]);

        input_permute  = std::stoi(argv[11]);
        output_permute = std::stoi(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        printf("arg11 to 12: input / output permute\n");
        exit(0);
    }

    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // A layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // A layout [G0, G1, M, K]

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, KV_head, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * KV_head * K, K, KV_head * K, 1}
            // B0 layout [G0, N, G1, K]
            : std::vector<ck::index_t>{KV_head * N * K, N * K, K, 1}; // B0 layout [G0, G1, N, K]

    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, KV_head, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * KV_head * O, O, 1, KV_head * O}
            // B1 layout [G0, N, G1, O]
            : std::vector<ck::index_t>{KV_head * N * O, N * O, 1, O}; // B1 layout [G0, G1, N, O]

    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // C layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // C layout [G0, G1, M, O]

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    case 4: // A, B0, B1 1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 5: // Rand: b1 b0; unit: a
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 6: // Rand: a b0 ; unit: B1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 7: // Rand: a b1 ; unit: b0
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 8: // Rand: a ; unit: b0 b1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 9: // Rand: b0 ; unit: a b1
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 10: // Rand: b1 ; unit: a b0
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    float best_perf         = .0;
    float best_time         = .0;
    int not_pass            = 0;
    std::string best_kernel = "";
    printf("Verification: %s\n", do_verification ? "ON" : "OFF");
    // TODO ANT: replace array with vector?
    ck::static_for<0, std::tuple_size_v<DeviceMHAFactory>, 1>{}([&](auto i) -> void {
        const auto device_mha_instance = std::get<i>(DeviceMHAFactory{});

        using DeviceMHAInstance = ck::remove_cvref_t<decltype(device_mha_instance)>;
        auto gemm               = DeviceMHAInstance{};
        auto invoker            = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                          static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
                                          static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
                                          static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          O,
                                          G0,
                                          G1,
                                          alpha,
                                          input_permute,
                                          output_permute);

        if(!gemm.IsSupportedArgument(argument))
        {
            std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

            // return 0;
        }

        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * G0 * G1;
        std::size_t num_btype =
            (sizeof(ADataType) * M * K + sizeof(CDataType) * M * O) * G0 * G1 +
            (sizeof(B0DataType) * K * N + sizeof(B1DataType) * N * O) * G0 * QueryGroupNumber;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
        if(tflops > best_perf)
        {
            best_perf   = tflops;
            best_time   = ave_time * 1000;
            best_kernel = gemm.GetTypeString();
        }
        if(do_verification)
        {
            c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

            Tensor<ADataType> a_g0_g1_m_k({G0, G1, M, K});
            Tensor<B0DataType> b0_g0_gq_k_n({G0, QueryGroupNumber, K, N});
            Tensor<B1DataType> b1_g0_gq_n_o({G0, QueryGroupNumber, N, O});
            Tensor<Acc0DataType> acc0_g0_g1_m_n({G0, G1, M, N}); // scratch object after gemm0
            Tensor<ADataType> a1_g0_g1_m_n({G0, G1, M, N});      // scratch object after softmax
            Tensor<CDataType> c_g0_g1_m_o_host_result({G0, G1, M, O}); // scratch object after gemm1

            // permute
            a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
                a_g0_g1_m_k(idx[0], idx[1], idx[2], idx[3]) = self(idx);
            });
            b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
                b0_g0_gq_k_n(idx[0], idx[1], idx[3], idx[2]) = self(idx);
            });
            b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
                b1_g0_gq_n_o(idx[0], idx[1], idx[3], idx[2]) = self(idx);
            });

            // gemm 0
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(a_g0_g1_m_k,
                                                             b0_g0_gq_k_n,
                                                             acc0_g0_g1_m_n,
                                                             a_element_op,
                                                             b0_element_op,
                                                             acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            // masking
            const auto mask = typename DeviceMHAInstance::C0MatrixMask(N);
            acc0_g0_g1_m_n.ForEach([&](auto& self, auto idx) {
                if(mask.IsMaskedElement(idx[2], idx[3]))
                    self(idx) = -ck::NumericLimits<float>::Infinity();
            });

            // softmax
            auto ref_softmax         = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker = ref_softmax.MakeInvoker();
            auto ref_softmax_argument =
                ref_softmax.MakeArgument(acc0_g0_g1_m_n, a1_g0_g1_m_n, 1, 0, {3});

            ref_softmax_invoker.Run(ref_softmax_argument);

            // gemm1
            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g0_g1_m_n,
                                                             b1_g0_gq_n_o,
                                                             c_g0_g1_m_o_host_result,
                                                             PassThrough{},
                                                             b1_element_op,
                                                             c_element_op);

            ref_gemm1_invoker.Run(ref_gemm1_argument);

            // permute
            c_gs_ms_os_host_result.ForEach(
                [&](auto& self, auto idx) { self(idx) = c_g0_g1_m_o_host_result(idx); });

            // default absolute error and relative error is 0.001
            double rtol = 1e-3;
            double atol = 1e-3;

            // when BF16 is taken, set absolute error and relative error to 0.01
            if(std::is_same_v<ADataType, ck::bhalf_t> && std::is_same_v<B0DataType, ck::bhalf_t> &&
               std::is_same_v<B1DataType, ck::bhalf_t> && std::is_same_v<CDataType, ck::bhalf_t>)
            {
                rtol = 1e-2;
                atol = 1e-2;
            }

            bool this_run_verification = ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                                              c_gs_ms_os_host_result.mData,
                                                              "Error: Incorrect results!",
                                                              rtol,
                                                              atol);
            printf("Verification: %s, Pass: %s\n",
                   do_verification ? "ON" : "OFF",
                   this_run_verification ? "YES" : "NO");

            if(!this_run_verification)
            {
                not_pass = 1;
                printf("%d th MQA instance verification Failed \n", i.value);
            }
        }
    });
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------"
              << std::endl;
    std::cout << "Problem Size: BatchCount: " << G0 << ", HeadNum: " << G1 << ", M: " << M
              << ", N: " << N << ", K: " << K << ", O: " << O << std::endl;
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------"
              << std::endl;
    std::cout << "Best kernel: " << best_kernel << " , " << best_perf << " TFlops , " << best_time
              << " us" << std::endl;
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------"
              << std::endl;
    return not_pass;
}
