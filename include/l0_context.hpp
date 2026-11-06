#pragma once
// Minimal RAII wrappers around the Level Zero loader.
//
// Everything throws std::runtime_error on failure: a Goldbach verifier must
// fail closed, never silently continue with half-created GPU state.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <ze_api.h>

#define ZE_CHECK(call)                                                        \
    do {                                                                      \
        ze_result_t _r = (call);                                              \
        if (_r != ZE_RESULT_SUCCESS) {                                        \
            throw std::runtime_error(std::string(#call) + " failed: " +       \
                                     gb::ze_result_string(_r));               \
        }                                                                     \
    } while (0)

namespace gb {

/// Human-readable name of a Level Zero result code (for error messages).
/// @param r  any ze_result_t, including codes not listed explicitly.
/// @return   a short description, or "error <n>" for unlisted codes.
std::string ze_result_string(ze_result_t r);

/// Device + context + in-order immediate command list for the first Level
/// Zero GPU. Arc B-Series supports immediate command lists; that is the mode
/// we use. Non-copyable; destroys the list and context on destruction.
class Context {
public:
    /// Initializes Level Zero and binds the first GPU device of the first
    /// driver that has one.
    /// @param unordered  create the list without IN_ORDER (dense pipeline
    ///        only, which then orders every launch by a barrier; sets
    ///        g_unordered). Other callers rely on in-order fills / copies.
    /// @post  driver(), device(), context() and list() are valid handles; the
    ///        list is IN_ORDER unless @p unordered (load-bearing, see
    ///        l0_context.cpp).
    /// @throws std::runtime_error if no driver / GPU exists or any call fails.
    explicit Context(bool unordered = false);
    ~Context();
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    ze_driver_handle_t  driver()  const { return driver_; }
    ze_device_handle_t  device()  const { return device_; }
    ze_context_handle_t context() const { return context_; }
    ze_command_list_handle_t list() const { return list_; }

    /// Allocates device-only USM (not host-accessible). The caller owns the
    /// memory and releases it with free_mem().
    /// @param bytes  requested size; sizes >= 1 MB are rounded up to a whole
    ///               number of 2 MB pages (2 MB aligned) for TLB reach.
    /// @param align  minimum alignment in bytes (power of two).
    /// @return       a non-null device pointer.
    /// @throws std::runtime_error on allocation failure.
    void* alloc_device(size_t bytes, size_t align = 64) const;
    /// Allocates shared (host + device accessible) USM; caller-owned, released
    /// with free_mem().
    /// @param bytes  size in bytes.
    /// @param align  minimum alignment in bytes (power of two).
    /// @return       a non-null pointer usable on host and device.
    /// @throws std::runtime_error on allocation failure.
    void* alloc_shared(size_t bytes, size_t align = 64) const;
    /// Frees USM from alloc_device / alloc_shared. A null @p ptr is a no-op.
    /// @pre  no enqueued GPU work still references @p ptr.
    static void free_mem(ze_context_handle_t ctx, void* ptr);

    /// Blocks until every command appended to the immediate list has finished.
    /// @throws std::runtime_error on synchronization failure (e.g. device lost).
    void host_sync() const;

private:
    ze_driver_handle_t  driver_  = nullptr;
    ze_device_handle_t  device_  = nullptr;
    ze_context_handle_t context_ = nullptr;
    ze_command_list_handle_t list_ = nullptr;
};

/// SPIR-V module loaded from a file (produced from kernels/*.cl by ocloc) and
/// JIT-compiled for the context's device. Non-copyable.
class Module {
public:
    /// @param ctx       context whose device the module is built for; must
    ///                  outlive the Module.
    /// @param spv_path  path to a non-empty SPIR-V file.
    /// @throws std::runtime_error if the file is missing/empty or the build fails.
    Module(const Context& ctx, const std::string& spv_path);
    ~Module();
    Module(const Module&)            = delete;
    Module& operator=(const Module&) = delete;

    ze_module_handle_t handle() const { return module_; }

private:
    ze_context_handle_t ctx_ = nullptr;
    ze_module_handle_t  module_ = nullptr;
};

/// The immediate list is not in-order (Context(true): the dense pipeline,
/// unless GB_UNORDERED=0); Kernel::launch then appends a barrier first.
extern bool g_unordered;

/// One compiled kernel inside a Module. Non-copyable.
class Kernel {
public:
    /// @param mod   module containing the kernel; must outlive the Kernel.
    /// @param name  kernel entry-point name.
    /// @throws std::runtime_error if the module has no such kernel.
    Kernel(const Module& mod, const std::string& name);
    ~Kernel();
    Kernel(const Kernel&)            = delete;
    Kernel& operator=(const Kernel&) = delete;

    /// Sets a 1-D work-group size of @p x work-items.
    /// @throws std::runtime_error if the device rejects the size.
    void set_group_size(uint32_t x);
    /// Sets a pointer / scalar argument.
    /// @param index  argument position in the kernel signature.
    /// @param size   sizeof the argument (e.g. sizeof(ptr), 4, 8).
    /// @param value  address of the value; copied immediately.
    /// @throws std::runtime_error on an invalid index or size.
    void set_arg(uint32_t index, size_t size, const void* value);
    /// Sets a __local buffer argument: Level Zero allocates @p size bytes of
    /// SLM per work-group (no host pointer).
    /// @throws std::runtime_error on an invalid index or size.
    void set_local_arg(uint32_t index, size_t size);
    /// Appends a launch of @p gx work-groups (1-D grid) to @p list.
    /// @pre   all arguments are set; buffers stay alive until the launch completes.
    /// @param event  optional event signaled on completion (kernel timing).
    /// @post  the launch is enqueued, not finished (use Context::host_sync or the event).
    /// @throws std::runtime_error if the append fails.
    void launch(ze_command_list_handle_t list, uint32_t gx,
                ze_event_handle_t event = nullptr);
    /// As launch, but without the barrier: the launch may
    /// overlap earlier commands. @pre it shares no buffer with any command
    /// still in flight.
    void launch_overlap(ze_command_list_handle_t list, uint32_t gx,
                        ze_event_handle_t event = nullptr);

private:
    ze_kernel_handle_t kernel_ = nullptr;
    uint32_t group_x_ = 1;
};

} // namespace gb
