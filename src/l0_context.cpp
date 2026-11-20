#include <algorithm>
#include "l0_context.hpp"
#include "env.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace gb {

std::string ze_result_string(ze_result_t r) {
    switch (r) {
    case ZE_RESULT_SUCCESS: return "success";
    case ZE_RESULT_NOT_READY: return "not ready";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "device lost";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "module build failure";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE: return "module link failure";
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET: return "device requires reset";
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE: return "device in low power state";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: return "insufficient permissions";
    case ZE_RESULT_ERROR_NOT_AVAILABLE: return "not available";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "dependency unavailable";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "uninitialized";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "unsupported version";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "unsupported feature";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "invalid null handle";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE: return "handle object in use";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "invalid null pointer";
    case ZE_RESULT_ERROR_INVALID_SIZE: return "invalid size";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "unsupported size";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "unsupported alignment";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT: return "invalid synchronization object";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "invalid enumeration";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION: return "unsupported enumeration";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY: return "invalid native binary";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME: return "invalid global name";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME: return "invalid kernel name";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME: return "invalid function name";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION: return "invalid group size dimension";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION: return "invalid global width dimension";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX: return "invalid kernel arg index";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE: return "invalid kernel arg size";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE: return "invalid kernel attribute value";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED: return "module unlinked";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE: return "invalid command list type";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS: return "overlapping regions";
    case ZE_RESULT_ERROR_UNKNOWN: return "unknown error";
    default: return "error " + std::to_string(std::to_underlying(r));
    }
}

// Driver asked for the text of a failure: the first GPU's (set by
// gpu_devices). Atomic: ZE_CHECK runs on the host worker threads too.
static std::atomic<ze_driver_handle_t> g_error_driver{nullptr};

std::string ze_failure(const char* call, ze_result_t r) {
    std::string msg = std::string(call) + " failed: " + ze_result_string(r);
    if (ze_driver_handle_t d = g_error_driver.load(std::memory_order_relaxed)) {
        const char* desc = nullptr;
        if (zeDriverGetLastErrorDescription(d, &desc) == ZE_RESULT_SUCCESS && desc && *desc)
            msg += std::string(" (driver: ") + desc + ")";
    }
    return msg;
}

std::vector<Gpu> gpu_devices() {
    ze_init_driver_type_desc_t idesc{};
    idesc.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
    idesc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
    uint32_t nd = 0;
    ZE_CHECK(zeInitDrivers(&nd, nullptr, &idesc));
    if (nd == 0) throw std::runtime_error("no Level Zero GPU driver found");
    std::vector<ze_driver_handle_t> drivers(nd);
    ZE_CHECK(zeInitDrivers(&nd, drivers.data(), &idesc));
    drivers.resize(nd);

    std::vector<Gpu> gpus;
    for (ze_driver_handle_t d : drivers) {
        uint32_t nv = 0;
        ZE_CHECK(zeDeviceGet(d, &nv, nullptr));
        std::vector<ze_device_handle_t> devs(nv);
        ZE_CHECK(zeDeviceGet(d, &nv, devs.data()));
        devs.resize(nv);
        for (ze_device_handle_t dev : devs) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ZE_CHECK(zeDeviceGetProperties(dev, &props));
            if (props.type == ZE_DEVICE_TYPE_GPU) gpus.push_back({d, dev});
        }
    }
    if (gpus.empty()) throw std::runtime_error("no GPU device reported by Level Zero");
    g_error_driver.store(gpus.front().driver, std::memory_order_relaxed);
    return gpus;
}

std::optional<ze_kernel_timestamp_data_t> kernel_ticks(ze_event_handle_t ev) {
    ze_kernel_timestamp_result_t ts{};
    if (zeEventQueryKernelTimestamp(ev, &ts) != ZE_RESULT_SUCCESS) return std::nullopt;
    return ts.context;
}

std::optional<double> kernel_seconds(ze_event_handle_t ev, double timer_hz) {
    if (!(timer_hz > 0)) return std::nullopt;
    const auto t = kernel_ticks(ev);
    if (!t) return std::nullopt;
    const int64_t d = int64_t(t->kernelEnd) - int64_t(t->kernelStart);
    if (d <= 0) return std::nullopt;
    return double(d) / timer_hz;
}

uint32_t env_grid(const char* name, uint32_t dflt) {
    return env::grid(name, dflt);
}

// ---------------------------------------------------------------- Context --

// EU thread arbitration hint for every Kernel: round-robin measured +0.6%
// over the Xe2 default (stall-based round-robin), oldest-first -0.6%; mixing
// policies between the kernels of a segment loses (HISTORY.md §9.2w).
// GB_SCHED=0 oldest-first, 1 round-robin (default),
// 2 stall-based, off = the driver's default. Parsed once per Context, before
// any kernel exists. Support is decided by zeKernelSchedulingHintExp itself
// (Kernel::Kernel), NOT by ze_scheduling_hint_exp_properties_t: the B580
// driver (26.40) reports no supported hint flags there although it accepts
// and applies round-robin (gating on the query would lose the +0.6%). A
// policy GB_SCHED asked for must be accepted (else an error);
// the default falls back to the driver's arbitration with a warning. The
// policy in force is reported by sched_policy_name() (run output: sched=).
static void sched_setup() {
    const std::string sv = env::choice("GB_SCHED", "1", {"0", "1", "2", "off"}, false);
    SchedPolicy p;
    p.requested = env::present("GB_SCHED");
    if (sv == "off") {
        p.name = "driver";
        g_sched = p;
        return;
    }
    if (sv == "0")      { p.flag = ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST; p.name = "oldest-first"; }
    else if (sv == "1") { p.flag = ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN; p.name = "round-robin"; }
    else if (sv == "2") { p.flag = ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN; p.name = "stall-based"; }
    else throw std::logic_error("GB_SCHED: unhandled value " + sv);   // env::choice checked it

    g_sched = p;
}

Context::Context(bool unordered) {
    const Gpu gpu = gpu_devices().front();
    driver_ = gpu.driver;
    device_ = gpu.device;

    sched_setup();
    // GB_UNORDERED=0: in-order list (dense pipeline only). Read before any Level Zero object is
    // created, so a malformed value throws without leaking one.
    const bool unordered_env = unordered && env::onoff("GB_UNORDERED", true);

    ze_context_desc_t cdesc{};
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ZE_CHECK(zeContextCreate(driver_, &cdesc, &context_));

    // Immediate command list: the supported/simplified submission mode on
    // Arc B-Series. Kernels appended here go straight toward execution.
    // IN_ORDER is load-bearing for correctness: the pipeline interleaves
    // kernels that share buffers without event dependencies between them
    // (sieve -> large primes -> check, and the two-slot buffer reuse across
    // segments). Without the flag, Level Zero gives NO
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
        if (unordered_env) { qdesc.flags = 0; g_unordered = true; }
    }
    qdesc.mode   = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreateImmediate(context_, device_, &qdesc, &list_));
}

Context::~Context() {
    if (list_)    zeCommandListDestroy(list_);
    if (context_) zeContextDestroy(context_);
}

void* Context::alloc_device(size_t bytes, size_t align, bool uncompressed) const {
    void* ptr = nullptr;
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ze_memory_compression_hints_ext_desc_t ch{};
    if (uncompressed) {
        ch.stype = ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC;
        ch.flags = ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED;
        desc.pNext = &ch;
    }
    // Large buffers are rounded up to a whole number of 2 MB pages (2 MB
    // aligned). Measured on the B580 (runtime 26.x): an allocation whose size
    // is not a 2 MB multiple is only partly backed by 2 MB pages, and the
    // large-prime kernel's scattered atomics over the ~15.6 MB prime map ran
    // 15-200% slower depending on where the map landed (TLB reach) — one
    // pipeline slot was consistently ~55% slower than the other. Rounded,
    // both slots run at the fast rate (large_s -30%). The tail is unused.
    constexpr size_t kHuge = 2uz << 20;
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

std::string locate_module(const std::string& path, bool exe_first) {
    namespace fs = std::filesystem;
    const auto readable = [](const fs::path& p) { return std::ifstream(p, std::ios::binary).good(); };
    if (path.empty()) throw std::runtime_error("no SPIR-V module path");
    std::vector<fs::path> cand;
    if (fs::path(path).is_relative()) {
        std::error_code ec;
        const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
        if (!ec) cand = {exe.parent_path() / path, exe.parent_path().parent_path() / path};
    }
    cand.insert(exe_first ? cand.end() : cand.begin(), fs::path(path));
    std::string tried;
    for (const fs::path& c : cand) {
        if (readable(c)) return c.string();
        tried += (tried.empty() ? "" : ", ") + c.string();
    }
    throw std::runtime_error("cannot open SPIR-V file: " + path + " (tried " + tried + ")");
}

Module::Module(const Context& ctx, const std::string& spv_path_arg) : ctx_(ctx.context()) {
    const std::string spv_path = locate_module(spv_path_arg);
    std::ifstream f(spv_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open SPIR-V file: " + spv_path);
    std::vector<char> data((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (data.empty()) throw std::runtime_error("empty SPIR-V file: " + spv_path);

    ze_module_desc_t desc{};
    desc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
    // A native ocloc binary (*.bin, e.g. built with -force_stos_opt) loads
    // as is; anything else is SPIR-V compiled by the driver.
    const bool native = spv_path.size() > 4 && spv_path.compare(spv_path.size() - 4, 4, ".bin") == 0;
    desc.format       = native ? ZE_MODULE_FORMAT_NATIVE : ZE_MODULE_FORMAT_IL_SPIRV;
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
    // EU thread arbitration (policy parsed and support-checked once by the
    // Context, see sched_setup).
    if (g_sched.flag == 0) return;
    ze_scheduling_hint_exp_desc_t hint{};
    hint.stype = ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC;
    hint.flags = g_sched.flag;
    const ze_result_t r = zeKernelSchedulingHintExp(kernel_, &hint);
    if (r == ZE_RESULT_SUCCESS) return;
    if (g_sched.requested) {
        // The destructor does not run for a throwing constructor.
        zeKernelDestroy(kernel_);
        kernel_ = nullptr;
        throw std::runtime_error("zeKernelSchedulingHintExp(" + name + ") failed: " +
                                 ze_result_string(r) + " (GB_SCHED=" + g_sched.name + ")");
    }
    // Default policy: keep going on the driver's arbitration, but say so
    // (once) and report it in the run output.
    if (!g_sched.failed)
        std::fprintf(stderr, "warning: EU scheduling hint %s rejected for %s (%s); "
                     "using the driver default (set GB_SCHED=off to silence)\n",
                     g_sched.name, name.c_str(), ze_result_string(r).c_str());
    g_sched.failed = true;
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
SchedPolicy g_sched;

const char* sched_policy_name() {
    return g_sched.failed ? "driver(hint-failed)" : g_sched.name;
}

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
