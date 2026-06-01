#pragma once

#include <cstdint>
#include <limits>
#include <utility>

namespace bag {

using VertexId = std::uint32_t;
using EdgeWeight = std::uint32_t;
using ObjId = std::uint32_t;
using SgId = std::size_t;
using Edge = std::pair<VertexId, VertexId>;

constexpr VertexId kInvalidVertex = 0;
constexpr EdgeWeight kInfWeight = std::numeric_limits<EdgeWeight>::max();

struct HalfWeight {
    EdgeWeight whole{0};
    bool half{false};

    [[nodiscard]] double to_double() const {
        return static_cast<double>(whole) + (half ? 0.5 : 0.0);
    }

    auto as_tuple() const {
        return std::make_pair(whole, half);
    }
};

inline bool operator==(const HalfWeight& lhs, const HalfWeight& rhs) {
    return lhs.whole == rhs.whole && lhs.half == rhs.half;
}

inline bool operator<(const HalfWeight& lhs, const HalfWeight& rhs) {
    return lhs.as_tuple() < rhs.as_tuple();
}

inline bool operator>(const HalfWeight& lhs, const HalfWeight& rhs) {
    return rhs < lhs;
}

inline bool operator<=(const HalfWeight& lhs, const HalfWeight& rhs) {
    return !(rhs < lhs);
}

inline bool operator>=(const HalfWeight& lhs, const HalfWeight& rhs) {
    return !(lhs < rhs);
}

inline Edge ordered_edge(VertexId u, VertexId v) {
    return (u < v) ? Edge{u, v} : Edge{v, u};
}

inline std::uint64_t pack_pair(VertexId u, VertexId v) {
    return (static_cast<std::uint64_t>(u) << 32U) | static_cast<std::uint64_t>(v);
}

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& value) const noexcept {
        return std::hash<T1>{}(value.first) ^ (std::hash<T2>{}(value.second) << 1U);
    }
};

}  // namespace bag
