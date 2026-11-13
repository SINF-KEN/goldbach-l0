#include <algorithm>
#include "l0_context.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace gb {

std::string ze_result_string(ze_result_t r) {
    switch (r) {
    case ZE_RESULT_SUCCESS: return "success";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "device lost";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "invalid null pointer";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "uninitialized";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "unsupported feature";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "module build failure";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX: return "invalid kernel arg index";
    case ZE_RESULT_ERROR_INVALID_SIZE: return "invalid size";
    default: return "error " + std::to_string(static_cast<int>(r));
    }
}

uint32_t env_grid(const char* name, uint32_t dflt) {
    const char* e = std::getenv(name);
    if (!e) return dflt;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(e, &end, 0);
    if (*e == '\0' || *end != '\0' || errno != 0 || v < 1 || v > 0xFFFFFFFFull)
        throw std::runtime_error(std::string(name) + "='" + e +
                                 "': expected a work-group count >= 1");
    return uint32_t(v);
}

// ---------------------------------------------------------------- Context --

Context::Context(bool unordered) {
    ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count = 0;
    ZE_CHECK(zeDriverGet(&driver_count, nullptr));
    if (driver_count == 0) throw std::runtime_error("no Level Zero driver found");
    std::vector<ze_driver_handle_t> drivers(driver_count);
    ZE_CHECK(zeDriverGet(&driver_count, drivers.data()));

    // Pick the first driver that has a GPU device.
    for (ze_driver_handle_t d : drivers) {
        uint32_t dev_count = 0;
        if (zeDeviceGet(d, &dev_count, nullptr) != ZE_RESULT_SUCCESS || dev_count == 0) continue;
        std::vector<ze_device_handle_t> devs(dev_count);
        ZE_CHECK(zeDeviceGet(d, &dev_count, devs.data()));
        for (ze_device_handle_t dev : devs) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ZE_CHECK(zeDeviceGetProperties(dev, &props));
            if (props.type == ZE_DEVICE_TYPE_GPU) {
                driver_ = d;
                device_ = dev;
                break;
            }
        }
        if (device_) break;
    }
    if (!device_) throw std::runtime_error("no GPU device found");

    ze_context_desc_t cdesc{};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ZE_CHECK(zeContextCreate(driver_, &cdesc, &context_));

    // Immediate command list: the supported/simplified submission mode on
    // Arc B-Series. Kernels appended here go straight toward execution.
    // IN_ORDER is load-bearing for correctness: the pipeline interleaves
    // kernels that share buffers without event dependencies between them
    // (prep_large_fb -> sieve -> large primes -> check, and the two-slot
    // buffer reuse across segments). Without the flag, Level Zero gives NO
    // ordering guarantee between appended commands, and e.g. the tile
    // kernel's whole-word map stores can race the large-prime kernel's
    // atomic ANDs (lost composite marks -> false verification).
    ze_command_queue_desc_t qdesc{};
    qdesc.stype  = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.flags  = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    // Dense pipeline (run_gpu, unless GB_UNORDERED=0): no implicit
    // ordering, but every Kernel::launch appends a full barrier first, so the
    // order above still holds for every launch except Kernel::launch_overlap
    // (the next segment's tile sieve and the hit-list harvest, which touch
    // only their own slot / the lists): they fill the GPU while the previous
    // segment's checker drains (+1.2%). Raw fills / copies get no barrier:
    // run_gpu orders the ones that need it itself.
    if (unordered) {
        const char* u = std::getenv("GB_UNORDERED");
        if (!(u && std::strcmp(u, "0") == 0)) { qdesc.flags = 0; g_unordered = true; }
    }
    qdesc.mode   = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreateImmediate(context_, device_, &qdesc, &list_));
}

Context::~Context() {
    if (list_)    zeCommandListDestroy(list_);
    if (context_) zeContextDestroy(context_);
}

void* Context::alloc_device(size_t bytes, size_t align) const {
    void* ptr = nullptr;
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    // Large buffers are rounded up to a whole number of 2 MB pages (2 MB
    // aligned). Measured on the B580 (runtime 26.x): an allocation whose size
    // is not a 2 MB multiple is only partly backed by 2 MB pages, and the
    // large-prime kernel's scattered atomics over the ~15.6 MB prime map ran
    // 15-200% slower depending on where the map landed (TLB reach) — one
    // pipeline slot was consistently ~55% slower than the other. Rounded,
    // both slots run at the fast rate (large_s -30%). The tail is unused.
    constexpr size_t kHuge = size_t(2) << 20;
    if (bytes >= kHuge / 2) {
        bytes = (bytes + kHuge - 1) / kHuge * kHuge;
        align = std::max(align, kHuge);
    }
    ZE_CHECK(zeMemAllocDevice(context_, &desc, bytes, align, device_, &ptr));
    return ptr;
}

void* Context::alloc_shared(size_t bytes, size_t align) const {
    void* ptr = nullptr;
    ze_host_mem_alloc_desc_t hdesc{};
    hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    ze_device_mem_alloc_desc_t ddesc{};
    ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ZE_CHECK(zeMemAllocShared(context_, &ddesc, &hdesc, bytes, align, device_, &ptr));
    return ptr;
}

void Context::free_mem(ze_context_handle_t ctx, void* ptr) {
    if (ptr) zeMemFree(ctx, ptr);
}

void Context::host_sync() const {
    ZE_CHECK(zeCommandListHostSynchronize(list_, UINT64_MAX));
}

// ----------------------------------------------------------------- Module --

Module::Module(const Context& ctx, const std::string& spv_path) : ctx_(ctx.context()) {
    std::ifstream f(spv_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open SPIR-V file: " + spv_path);
    std::vector<char> data((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (data.empty()) throw std::runtime_error("empty SPIR-V file: " + spv_path);

    ze_module_desc_t desc{};
    desc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format       = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.inputSize    = data.size();
    desc.pInputModule = reinterpret_cast<const uint8_t*>(data.data());
    ZE_CHECK(zeModuleCreate(ctx.context(), ctx.device(), &desc, &module_, nullptr));
}

Module::~Module() {
    if (module_) zeModuleDestroy(module_);
}

// ----------------------------------------------------------------- Kernel --

Kernel::Kernel(const Module& mod, const std::string& name) {
    ze_kernel_desc_t desc{};
    desc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = name.c_str();
    ZE_CHECK(zeKernelCreate(mod.handle(), &desc, &kernel_));
}

Kernel::~Kernel() {
    if (kernel_) zeKernelDestroy(kernel_);
}

void Kernel::set_group_size(uint32_t x) {
    group_x_ = x;
    ZE_CHECK(zeKernelSetGroupSize(kernel_, x, 1, 1));
}

void Kernel::set_arg(uint32_t index, size_t size, const void* value) {
    ZE_CHECK(zeKernelSetArgumentValue(kernel_, index, size, value));
}

void Kernel::set_local_arg(uint32_t index, size_t size) {
    // nullptr + size => Level Zero allocates the __local (SLM) buffer.
    ZE_CHECK(zeKernelSetArgumentValue(kernel_, index, size, nullptr));
}

bool g_unordered = false;

void Kernel::launch(ze_command_list_handle_t list, uint32_t gx,
                    ze_event_handle_t event) {
    if (g_unordered) ZE_CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    launch_overlap(list, gx, event);
}

void Kernel::launch_overlap(ze_command_list_handle_t list, uint32_t gx,
                            ze_event_handle_t event) {
    ze_group_count_t count{gx, 1, 1};
    ZE_CHECK(zeCommandListAppendLaunchKernel(list, kernel_, &count, event,
                                             0, nullptr));
}

} // namespace gb
