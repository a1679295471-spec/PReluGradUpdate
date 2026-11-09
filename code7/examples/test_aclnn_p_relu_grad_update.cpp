#include "sample_support.h"
#include "aclnn_p_relu_grad_update.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({4, 6, 8}, ACL_FLOAT, 768, "grads.bin", true, false, root);
            Tensor input_1_0({4, 6, 8}, ACL_FLOAT, 768, "features.bin", true, false, root);
            Tensor input_2_0({6}, ACL_FLOAT, 24, "weights.bin", true, false, root);
            Tensor output_0_0({4, 6, 8}, ACL_FLOAT, 768, "dx.bin", false, false, root);
            Tensor output_1_0({4, 6, 8}, ACL_FLOAT, 768, "update.bin", false, false, root);
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnPReluGradUpdateGetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                input_2_0.tensor,
                output_0_0.tensor,
                output_1_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnPReluGradUpdate(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            output_1_0.save(root);
            if (workspace)
                check(aclrtFree(workspace));
        }
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(device));
        check(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
