#include "sample_support.h"
#include "aclnn_strided_slice_grad.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            // CANN 9.0's aclnn generator exposes a ValueDepend 1-D integer
            // input (`shape`) as aclIntArray, not aclTensor.
            const int64_t shapeValues[] = {2, 4, 5, 6};
            aclIntArray* shape = aclCreateIntArray(shapeValues, 4);
            if (shape == nullptr)
                throw std::runtime_error("shape array creation failed");
            Tensor input_1_0({4}, ACL_INT32, 16, "begin.bin", true, false, root);
            Tensor input_2_0({4}, ACL_INT32, 16, "end.bin", true, false, root);
            Tensor input_3_0({4}, ACL_INT32, 16, "strides.bin", true, false, root);
            Tensor input_4_0({2, 3, 5, 6}, ACL_FLOAT, 720, "dy.bin", true, false, root);
            Tensor output_0_0({2, 4, 5, 6}, ACL_FLOAT, 960, "y.bin", false, false, root);
            int64_t attr_0 = 0LL;
            int64_t attr_1 = 0LL;
            int64_t attr_2 = 0LL;
            int64_t attr_3 = 0LL;
            int64_t attr_4 = 0LL;
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnStridedSliceGradGetWorkspaceSize(
                shape,
                input_1_0.tensor,
                input_2_0.tensor,
                input_3_0.tensor,
                input_4_0.tensor,
                attr_0,
                attr_1,
                attr_2,
                attr_3,
                attr_4,
                output_0_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnStridedSliceGrad(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            if (workspace)
                check(aclrtFree(workspace));
            aclDestroyIntArray(shape);
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
