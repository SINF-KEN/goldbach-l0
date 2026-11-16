#pragma once
// Strict parsing of the GB_* environment variables (docs/PROJECT-REFERENCE.md
// §7). Every knob is read through these helpers so that a malformed value
// fails the run closed instead of silently becoming 0 or the default
// (docs/CORRECTNESS.md): non-numeric text, trailing junk, a sign, overflow,
// a value outside the knob's range and an unknown enum string all throw
// std::runtime_error naming the variable and what it accepts. main() turns
// that into "error: ..." and exit status 1, before any GPU work where the
// knob is read up front.
//
// Unset always means the default. Whether an EMPTY value means "unset" is
// per knob and kept as it was before these helpers: on/off switches and file
// paths treat "" as unset, numbers reject it (use is_set() where a number
// historically ignored ""), presence flags count it as present.
//
// Throwing helpers must not run where an exception cannot propagate (worker
// threads, destructors, noexcept functions): read the knob once up front.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>

namespace gb::env {

/// @return the raw value of @p name, or nullptr when unset.
inline const char* raw(const char* name) { return std::getenv(name); }

/// Presence flag: set to anything, including "" and "0".
inline bool present(const char* name) { return raw(name) != nullptr; }

/// Set to a non-empty value.
inline bool is_set(const char* name)
{
    const char* e = raw(name);
    return e && *e;
}

/// @throws std::runtime_error "NAME='value': expected <accepts>".
[[noreturn]] inline void reject(const char* name, const char* value, const std::string& accepts)
{
    throw std::runtime_error(std::string(name) + "='" + value + "': expected " + accepts);
}

/// Parses an unsigned integer like strtoull(s, &end, 0) -- decimal, 0x hex,
/// leading-0 octal -- but only when the WHOLE string is that number: no
/// leading whitespace or sign, no trailing characters, no overflow.
/// @return the value, or nullopt for anything else (including "").
inline std::optional<uint64_t> parse_u64(const char* s)
{
    if (!s || *s < '0' || *s > '9') return std::nullopt;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0') return std::nullopt;
    return uint64_t(v);
}

/// Parses a finite decimal floating-point number like strtod, the whole
/// string only (no leading whitespace, no trailing characters, no inf / nan,
/// no hex floats).
inline std::optional<double> parse_f64(const char* s)
{
    if (!s) return std::nullopt;
    const char* d = (*s == '-' || *s == '+') ? s + 1 : s;
    if (!((*d >= '0' && *d <= '9') || *d == '.')) return std::nullopt;
    if (d[0] == '0' && (d[1] == 'x' || d[1] == 'X')) return std::nullopt;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !std::isfinite(v)) return std::nullopt;
    return v;
}

inline std::string range_text(uint64_t lo, uint64_t hi)
{
    return "an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) +
           "] (decimal, 0x hex or 0-prefixed octal)";
}

/// @return @p name as an integer in [lo, hi], or @p dflt when unset.
/// @param what  replaces the generated "an integer in [lo, hi]" text.
/// @throws std::runtime_error when set but not such an integer ("" included).
inline uint64_t u64(const char* name, uint64_t dflt, uint64_t lo = 0, uint64_t hi = UINT64_MAX,
                    const char* what = nullptr)
{
    const char* e = raw(name);
    if (!e) return dflt;
    const auto v = parse_u64(e);
    if (!v || *v < lo || *v > hi) reject(name, e, what ? what : range_text(lo, hi));
    return *v;
}

/// u64 for a value that must fit 32 bits.
inline uint32_t u32(const char* name, uint32_t dflt, uint32_t lo = 0, uint32_t hi = UINT32_MAX,
                    const char* what = nullptr)
{
    return uint32_t(u64(name, dflt, lo, hi, what));
}

/// @return @p name as a finite number in [lo, hi], or @p dflt when unset.
inline double f64(const char* name, double dflt, double lo, double hi)
{
    const char* e = raw(name);
    if (!e) return dflt;
    const auto v = parse_f64(e);
    if (!v || !(*v >= lo && *v <= hi))
        reject(name, e, "a finite decimal number in [" + std::to_string(lo) + ", " +
                        std::to_string(hi) + "]");
    return *v;
}

/// On/off switch: unset or "" -> @p dflt, "0" -> false, "1" -> true.
/// @throws std::runtime_error for any other value.
inline bool onoff(const char* name, bool dflt)
{
    const char* e = raw(name);
    if (!e || !*e) return dflt;
    const std::string v = e;
    if (v == "0") return false;
    if (v == "1") return true;
    reject(name, e, "0 or 1");
}

/// Enumerated string: unset (or "" when @p empty_is_unset) -> @p dflt,
/// otherwise one of @p allowed exactly.
/// @throws std::runtime_error for any other value.
inline std::string choice(const char* name, const char* dflt,
                          std::initializer_list<const char*> allowed, bool empty_is_unset)
{
    const char* e = raw(name);
    if (!e || (empty_is_unset && !*e)) return dflt;
    std::string list;
    for (const char* a : allowed) {
        if (std::string(e) == a) return e;
        list += list.empty() ? "" : ", ";
        list += a;
    }
    reject(name, e, "one of: " + list);
}

/// File path / command: unset or "" -> @p dflt.
inline std::string path(const char* name, const char* dflt)
{
    const char* e = raw(name);
    return (e && *e) ? e : dflt;
}

/// Work-group size of a kernel compiled for SIMD32 subgroups: a non-zero
/// multiple of 32 up to 1024 (0 would divide the launch by zero, anything
/// else splits a subgroup).
inline uint32_t wg32(const char* name, uint32_t dflt)
{
    const uint32_t v = u32(name, dflt, 32, 1024, "a multiple of 32 in [32, 1024]");
    if (v % 32 != 0) reject(name, raw(name), "a multiple of 32 in [32, 1024]");
    return v;
}

/// Grid-stride work-group count: >= 1 (a zero grid would skip the work and
/// still PASS). @p dflt may be 0 to mean "unset" to the caller.
inline uint32_t grid(const char* name, uint32_t dflt)
{
    return u32(name, dflt, 1, UINT32_MAX, "a work-group count >= 1");
}

} // namespace gb::env
