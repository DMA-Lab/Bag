//
// Created by sunnysab on 23-6-26.
//

#ifndef BAP_ALGORITHM_EXCEPTION_H
#define BAP_ALGORITHM_EXCEPTION_H


#include <stdexcept>
#include <utility>

class Exception : public std::exception {
    std::string message;

public:
    explicit Exception(std::string  message) : message(std::move(message)) {}

    const char *what() const noexcept override {
        return message.c_str();
    }
};


#endif //BAP_ALGORITHM_EXCEPTION_H
