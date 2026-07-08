//
// Created by sunnysab on 23-6-21.
//

#ifndef BAP_ALGORITHM_BITSET_H
#define BAP_ALGORITHM_BITSET_H

#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>

/// 由于标准库提供的 bitset 必须在编译期预先设定长度，
/// 这里单独实现了一个 bitset 类，其主要目的是节约标志位存储的内存。
/// 如果考虑到 CPU cache, 它可能会加快标志访问的速度。
class bitset {
    size_t capacity;
    uint8_t *data = nullptr;

public:
    bitset(unsigned int capacity) {
        auto actually_allocation = static_cast<size_t>(ceil(capacity / 8.0));
        this->capacity = actually_allocation;
        this->data = new uint8_t[actually_allocation];
    }

    ~bitset() {
        if (nullptr != this->data) {
            delete[] this->data;
            this->data = nullptr;
        }
    }

    void set(size_t index) {
        // TODO: 编译器可能会将下面两个计算替换成移位和减法
        // 暂时不优化
        int i = index / 8;
        int r = index % 8;
        if (nullptr != this->data) {
            this->data[i] |= (1 << r);
        }
    }

    bool get(size_t index) {
        // 同 set 函数
        int i = index / 8;
        int r = index % 8;
        if (nullptr != this->data) {
            return this->data[i] & (1 << r);
        }
        throw std::runtime_error("bitset inner array is null.");
    }

    void clear() {
        if (nullptr != this->data) {
            memset(this->data, 0, this->capacity);
        }
    }
};

#endif //BAP_ALGORITHM_BITSET_H
