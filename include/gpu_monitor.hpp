#pragma once
// GPU power / frequency / throttling over a run, through Level Zero Sysman
// (zes_api.h). Observation only: nothing here can fail or slow a run — any
// Sysman call that is unsupported or not permitted just leaves its value
// unmeasured, and the reason is reported.
//
// What hwmon (ci/gbci/report.py, scripts/goldbach_live.py) cannot say is WHY
// a run was slow: GpuMonitor records the hardware throttle-time counter, the
// throttle reasons seen and the actual core frequency, next to the energy.

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace gb {

/// What GpuMonitor measured over one start()..stop() window. A value is
/// meaningful only when its `have_*` flag is set.
struct GpuPower {
    std::string status = "off";      ///< "ok", "off" (GB_SYSMAN=0) or why Sysman is unusable
    double window_s = 0.0;           ///< wall time of the window (host clock)

    bool   have_energy = false;
    double energy_j    = 0.0;        ///< card (else first) power domain, joules
    double power_w     = 0.0;        ///< energy_j / the counter's own elapsed time

    bool   have_throttle = false;
    double throttled_s   = 0.0;      ///< hardware throttle-time counter delta (GPU domain)

    bool     have_freq   = false;
    double   freq_avg_mhz = 0.0;     ///< mean of the actual-frequency samples (~4 per second)
    double   freq_min_mhz = 0.0;
    uint32_t throttle_reasons = 0;   ///< OR of zes_freq_throttle_reason_flags_t seen in samples
    uint32_t samples = 0;

    /// Reasons as text, e.g. "thermal+power", or "none".
    std::string reasons_text() const;
};

/// Samples Sysman on a background thread between start() and stop().
/// GB_SYSMAN=0 disables it (status "off"). Sysman failures never throw.
/// @throws std::runtime_error (constructor only) on a malformed GB_SYSMAN.
class GpuMonitor {
public:
    GpuMonitor();
    ~GpuMonitor();
    GpuMonitor(const GpuMonitor&)            = delete;
    GpuMonitor& operator=(const GpuMonitor&) = delete;

    /// Takes the opening counter snapshots and starts the frequency sampler.
    void start();
    /// Stops the sampler and returns the window's measurements.
    GpuPower stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::thread sampler_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
    void sample_loop();
};

} // namespace gb
