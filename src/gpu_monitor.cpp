#include "gpu_monitor.hpp"
#include "env.hpp"
#include "l0_context.hpp"

#include <zes_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <format>
#include <vector>

namespace gb {

namespace {

using clock_ = std::chrono::steady_clock;

/// The Sysman device with the same UUID as the core GPU Context binds
/// (gpu_devices().front()); the first Sysman device when UUIDs are unavailable.
/// @return the device, or nullptr with @p why set.
zes_device_handle_t find_sysman_device(std::string& why) {
    if (zesInit(0) != ZE_RESULT_SUCCESS) { why = "zesInit failed"; return nullptr; }
    uint32_t nd = 0;
    if (zesDriverGet(&nd, nullptr) != ZE_RESULT_SUCCESS || nd == 0) { why = "no sysman driver"; return nullptr; }
    std::vector<zes_driver_handle_t> drivers(nd);
    if (zesDriverGet(&nd, drivers.data()) != ZE_RESULT_SUCCESS) { why = "zesDriverGet failed"; return nullptr; }

    ze_device_properties_t core{};
    core.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    bool have_uuid = false;
    try {
        have_uuid = zeDeviceGetProperties(gpu_devices().front().device, &core) == ZE_RESULT_SUCCESS;
    } catch (const std::exception&) {}

    zes_device_handle_t first = nullptr;
    for (zes_driver_handle_t d : drivers) {
        uint32_t nv = 0;
        if (zesDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS || nv == 0) continue;
        std::vector<zes_device_handle_t> devs(nv);
        if (zesDeviceGet(d, &nv, devs.data()) != ZE_RESULT_SUCCESS) continue;
        for (zes_device_handle_t dev : devs) {
            zes_device_ext_properties_t ext{};
            ext.stype = ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES;
            zes_device_properties_t p{};
            p.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            p.pNext = &ext;
            if (zesDeviceGetProperties(dev, &p) != ZE_RESULT_SUCCESS) continue;
            // A driver that ignores the ext struct leaves it zeroed: fall
            // back to the (deprecated) core properties for type and UUID.
            const bool ext_ok = ext.type == ZES_DEVICE_TYPE_GPU;
            if (!ext_ok && p.core.type != ZE_DEVICE_TYPE_GPU) continue;
            if (!first) first = dev;
            static_assert(sizeof ext.uuid.id == sizeof core.uuid.id);
            const uint8_t* id = ext_ok ? ext.uuid.id : p.core.uuid.id;
            if (have_uuid && std::memcmp(id, core.uuid.id, sizeof core.uuid.id) == 0)
                return dev;
        }
    }
    if (!first) why = "no sysman GPU";
    return first;
}

} // namespace

struct GpuMonitor::Impl {
    zes_device_handle_t dev = nullptr;
    zes_pwr_handle_t    pwr = nullptr;    // card domain, else the first one
    zes_freq_handle_t   freq = nullptr;   // GPU core domain
    std::string status = "off";

    clock_::time_point t0;
    bool have_e0 = false, have_t0 = false;
    zes_power_energy_counter_t e0{};
    zes_freq_throttle_time_t   th0{};

    // sampler results (guarded by GpuMonitor::mu_)
    double freq_sum = 0.0, freq_min = 0.0;
    uint32_t samples = 0, reasons = 0;

    void sample() {
        zes_freq_state_t st{};
        st.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;
        if (!freq || zesFrequencyGetState(freq, &st) != ZE_RESULT_SUCCESS) return;
        reasons |= st.throttleReasons;
        if (st.actual > 0) {
            freq_sum += st.actual;
            freq_min = samples ? std::min(freq_min, st.actual) : st.actual;
            ++samples;
        }
    }
};

GpuMonitor::GpuMonitor() : impl_(std::make_unique<Impl>()) {
    // A malformed GB_SYSMAN is an error like every GB_* knob (before any GPU work).
    if (!env::onoff("GB_SYSMAN", true)) return;   // status "off"
    std::string why;
    impl_->dev = find_sysman_device(why);
    if (!impl_->dev) { impl_->status = why; return; }

    if (zesDeviceGetCardPowerDomain(impl_->dev, &impl_->pwr) != ZE_RESULT_SUCCESS || !impl_->pwr) {
        impl_->pwr = nullptr;
        uint32_t n = 1;
        if (zesDeviceEnumPowerDomains(impl_->dev, &n, &impl_->pwr) != ZE_RESULT_SUCCESS || n == 0)
            impl_->pwr = nullptr;
    }
    uint32_t nf = 0;
    if (zesDeviceEnumFrequencyDomains(impl_->dev, &nf, nullptr) == ZE_RESULT_SUCCESS && nf) {
        std::vector<zes_freq_handle_t> fs(nf);
        if (zesDeviceEnumFrequencyDomains(impl_->dev, &nf, fs.data()) == ZE_RESULT_SUCCESS) {
            for (uint32_t i = 0; i < nf; ++i) {
                zes_freq_properties_t fp{};
                fp.stype = ZES_STRUCTURE_TYPE_FREQ_PROPERTIES;
                if (zesFrequencyGetProperties(fs[i], &fp) == ZE_RESULT_SUCCESS &&
                    fp.type == ZES_FREQ_DOMAIN_GPU && !fp.onSubdevice) {
                    impl_->freq = fs[i];
                    break;
                }
            }
        }
    }
    impl_->status = (impl_->pwr || impl_->freq) ? "ok" : "no power or frequency domain";
}

GpuMonitor::~GpuMonitor() {
    if (sampler_.joinable()) {
        { std::lock_guard lk(mu_); stopping_ = true; }
        cv_.notify_all();
        sampler_.join();
    }
}

void GpuMonitor::start() {
    Impl& m = *impl_;
    m.t0 = clock_::now();
    if (m.status != "ok") return;
    m.have_e0 = m.pwr && zesPowerGetEnergyCounter(m.pwr, &m.e0) == ZE_RESULT_SUCCESS;
    m.have_t0 = m.freq && zesFrequencyGetThrottleTime(m.freq, &m.th0) == ZE_RESULT_SUCCESS;
    if (m.freq) {
        try {
            sampler_ = std::thread([this] { sample_loop(); });
        } catch (const std::exception&) {}   // no sampler: frequency stays unmeasured
    }
}

void GpuMonitor::sample_loop() {
    std::unique_lock lk(mu_);
    while (!stopping_) {
        impl_->sample();
        cv_.wait_for(lk, std::chrono::milliseconds(250), [this] { return stopping_; });
    }
}

GpuPower GpuMonitor::stop() {
    Impl& m = *impl_;
    GpuPower r;
    r.status = m.status;
    r.window_s = std::chrono::duration<double>(clock_::now() - m.t0).count();
    if (sampler_.joinable()) {
        { std::lock_guard lk(mu_); stopping_ = true; }
        cv_.notify_all();
        sampler_.join();
    }
    if (m.status != "ok") return r;

    zes_power_energy_counter_t e1{};
    if (m.have_e0 && zesPowerGetEnergyCounter(m.pwr, &e1) == ZE_RESULT_SUCCESS &&
        e1.energy >= m.e0.energy && e1.timestamp > m.e0.timestamp) {
        r.have_energy = true;
        r.energy_j = double(e1.energy - m.e0.energy) * 1e-6;
        r.power_w  = r.energy_j / (double(e1.timestamp - m.e0.timestamp) * 1e-6);
    }
    zes_freq_throttle_time_t th1{};
    if (m.have_t0 && zesFrequencyGetThrottleTime(m.freq, &th1) == ZE_RESULT_SUCCESS &&
        th1.throttleTime >= m.th0.throttleTime) {
        r.have_throttle = true;
        r.throttled_s = double(th1.throttleTime - m.th0.throttleTime) * 1e-6;
    }
    std::lock_guard lk(mu_);
    if (m.samples) {
        r.have_freq = true;
        r.freq_avg_mhz = m.freq_sum / m.samples;
        r.freq_min_mhz = m.freq_min;
        r.samples = m.samples;
        r.throttle_reasons = m.reasons;
    }
    return r;
}

std::string GpuPower::reasons_text() const {
    static constexpr struct { uint32_t bit; const char* name; } kNames[] = {
        {ZES_FREQ_THROTTLE_REASON_FLAG_AVE_PWR_CAP,   "pl1"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_BURST_PWR_CAP, "pl2"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_CURRENT_LIMIT, "pl4"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL_LIMIT, "tjmax"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_PSU_ALERT,     "psu"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_SW_RANGE,      "sw-range"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_HW_RANGE,      "hw-range"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_VOLTAGE,       "voltage"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL,       "thermal"},
        {ZES_FREQ_THROTTLE_REASON_FLAG_POWER,         "power"},
    };
    std::string s;
    uint32_t rest = throttle_reasons;
    for (const auto& [bit, name] : kNames) {
        if (!(throttle_reasons & bit)) continue;
        if (!s.empty()) s += '+';
        s += name;
        rest &= ~bit;
    }
    if (rest) s += (s.empty() ? "" : "+") + std::string("0x") + std::format("{:x}", rest);
    return s.empty() ? "none" : s;
}

} // namespace gb
