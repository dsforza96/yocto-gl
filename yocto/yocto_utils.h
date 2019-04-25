//
// # Yocto/Utils: Tiny collection of utilities to support Yocto/GL
//
//
// Yocto/Utils is a collection of utilities used in writing other Yocto/GL
// libraries and example applications. We support printing builtin and
// Yocto/Math values, simple path manipulation, file lading/saving and basic
// concurrency utilities. These utilities are likely to change often and are to
// be considered internal to Yocto.
//
//
// ## Python-like iterators and collection helpers
//
// This library includes a set of functions to help use C++ collections with
// more ease, inspired by Python. All functions and operators are defined in
// the yocto namespace so they will not affect the code outside. But within
// the Yocto/GL collection they are the best way to do this.
//
// 1. use `range()` to iterato over an integer sequence
// 2. use `enumerate()` to iteratare over a vector and number its elements
// 3. use opeartors + to either concatenate two vectors or a vector and an
//    element
// 4. use operators += to append an element or a vector to a given vector
//
//
// ## Path manipulation
//
// We define a few path manipulation utilities to split and join path
// components.
//
// 1. Get paths components with `get_dirname()`, `get_filename()` and
//   `get_extension()`
// 2. Remove parts of a path with get_noextension() and `get_basename()`
// 3. check if a file exists with `exists_file()`
//
//
// ## File IO
//
// 1. load and save text files with `load_text()` and `save_text()`
// 2. load and save binary files with `load_binary()` and `save_binary()`
//
//
// ## Concurrency utilities
//
// C++ has very basic supprt for concurrency and most of it is still platform
// dependent. We provide here very basic support for concurrency utlities
// built on top of C++ low-level threading and synchronization.
//
// 1. use `concurrent_queue()` for communicationing values between threads
// 2. use `parallel_for()` for basic parallel for loops
//
//
// LICENSE:
//
// Copyright (c) 2016 -- 2019 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#ifndef _YOCTO_UTILS_H_
#define _YOCTO_UTILS_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include "yocto_math.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <deque>
#include <future>
#include <initializer_list>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define FMT_HEADER_ONLY
#include "ext/fmt/chrono.h"
#include "ext/fmt/format.h"

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

using std::atomic;
using std::deque;
using std::future;
using std::initializer_list;
using std::lock_guard;
using std::mutex;
using std::runtime_error;
using std::string;
using std::string_view;
using std::thread;
using std::vector;
using namespace std::string_literals;
using namespace std::string_view_literals;
using namespace std::chrono_literals;

using fmt::format;
using fmt::print;

}  // namespace yocto

// -----------------------------------------------------------------------------
// PRINTING AND FORMATTING HELPERS
// -----------------------------------------------------------------------------
namespace yocto {

// We use the fmt library as a backend for printing. These are just helpers to
// make printing easier in console apps.

// Helper to indicate info printing in console apps.
template <typename... Args>
inline void print_info(string_view format_str, const Args&... args);
// Prints an error and exit.
template <typename... Args>
inline void print_fatal(string_view format_str, const Args&... args);

// get time from a high resolution clock
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
inline time_point get_time();

// print information and returns a timer that will print the time when
// destroyed. Use with RIIA for scoped timing.
struct print_timer;
template <typename... Args>
inline print_timer print_timed(string_view format_str, const Args&... args);

// Logging to a sync
inline void set_log_callback(function<void(const string&)> callback);
template <typename... Args>
inline void log_info(string_view format_str, const Args&... args);
template <typename... Args>
inline void log_error(string_view format_str, const Args&... args);
struct log_timer;
template <typename... Args>
inline auto log_timed(string_view format_str, const Args&... args);

}  // namespace yocto

// Formatter for math types
namespace fmt {

// Formatter for math types
template <typename T, int N>
struct formatter<yocto::vec<T, N>>;
template <typename T, int N, int M>
struct formatter<yocto::mat<T, N, M>>;
template <typename T, int N>
struct formatter<yocto::affine<T, N>>;
template <typename T, int N>
struct formatter<yocto::frame<T, N>>;
template <typename T, int N>
struct formatter<yocto::bbox<T, N>>;

}  // namespace fmt

// -----------------------------------------------------------------------------
// PYTHON-LIKE ITERATORS
// -----------------------------------------------------------------------------
namespace yocto {

// Python `range()` equivalent. Construct an object to iterate over a sequence.
template <typename T>
struct range_iterator;
template <typename T>
inline range_iterator<T> range(T max);
template <typename T>
inline range_iterator<T> range(T min, T max);

// Python `enumerate()` equivalent. Construct an object that iteraterates over a
// sequence of elements and numbers them.
template <typename T>
struct enumerate_iterator;
template <typename T>
inline enumerate_iterator<const T> enumerate(const vector<T>& vals);
template <typename T>
inline enumerate_iterator<T> enumerate(vector<T>& vals);

// Vector append and concatenation
template <typename T>
inline vector<T>& operator+=(vector<T>& a, const T& b);
template <typename T>
inline vector<T>& operator+=(vector<T>& a, const vector<T>& b);
template <typename T>
inline vector<T>& operator+=(vector<T>& a, const initializer_list<T>& b);
template <typename T>
inline vector<T> operator+(const vector<T>& a, const T& b);
template <typename T>
inline vector<T> operator+(const vector<T>& a, const vector<T>& b);
template <typename T>
inline vector<T> operator+(const vector<T>& a, const initializer_list<T>& b);

}  // namespace yocto

// -----------------------------------------------------------------------------
// APPLICATION UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Format duration string from nanoseconds
inline string format_duration(int64_t duration);

}  // namespace yocto

// -----------------------------------------------------------------------------
// PATH UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Normalize path delimiters.
inline string normalize_path(const string& filename);
// Get directory name (including '/').
inline string get_dirname(const string& filename);
// Get extension (not including '.').
inline string get_extension(const string& filename);
// Get filename without directory.
inline string get_filename(const string& filename);
// Get path without extension.
inline string get_noextension(const string& filename);
// Get filename without directory and extension.
inline string get_basename(const string& filename);

// Check if a file can be opened for reading.
inline bool exists_file(const string& filename);

}  // namespace yocto

// -----------------------------------------------------------------------------
// FILE IO
// -----------------------------------------------------------------------------
namespace yocto {

// Load/save a text file
inline void load_text(const string& filename, string& str);
inline void save_text(const string& filename, const string& str);

// Load/save a binary file
inline void load_binary(const string& filename, vector<byte>& data);
inline void save_binary(const string& filename, const vector<byte>& data);

// Io error
struct io_error : runtime_error {
    explicit io_error(const char* msg) : runtime_error{msg} {}
    explicit io_error(const std::string& msg) : runtime_error{msg} {}
};

}  // namespace yocto

// -----------------------------------------------------------------------------
// CONCURRENCY UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// a simple concurrent queue that locks at every call
template <typename T>
struct concurrent_queue {
    concurrent_queue();
    concurrent_queue(const concurrent_queue& other);
    concurrent_queue& operator=(const concurrent_queue& other);

    bool empty();
    void clear();
    void push(const T& value);
    bool try_pop(T& value);

   private:
    mutex    _mutex;
    deque<T> _queue;
};

// Runs a rask as an asycnrhonous operation.
template <typename Function>
inline auto async(Function&& function);
template <typename Function, typename... Args>
inline auto async(Function&& function, Args&&... args);

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes the integer index.
template <typename Func>
inline void parallel_for(size_t begin, size_t end, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false);
template <typename Func>
inline void parallel_for(size_t num, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false);

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes a reference to a `T`.
template <typename T, typename Func>
inline void parallel_foreach(vector<T>& values, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false);
template <typename T, typename Func>
inline void parallel_foreach(const vector<T>& values, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false);

}  // namespace yocto

// ---------------------------------------------------------------------------//
//                                                                            //
//                             IMPLEMENTATION                                 //
//                                                                            //
// ---------------------------------------------------------------------------//

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PRINTING AND FORMATTING HELPERS
// -----------------------------------------------------------------------------
namespace yocto {

// Helper to indicate info printing in console apps.
template <typename... Args>
inline void print_info(string_view format_str, const Args&... args) {
    print(format_str, args...);
    print("\n");
}

// Prints an error and exit.
template <typename... Args>
inline void print_fatal(string_view format_str, const Args&... args) {
    print(format_str, args...);
    print("\n");
    exit(1);
}

// get time from a high resolution clock
inline time_point get_time() {
    return std::chrono::high_resolution_clock::now();
}

// print information and returns a timer that will print the time when
// destroyed. Use with RIIA for scoped timing.
struct print_timer {
    print_timer(const string& msg) : msg{msg} {
        start = get_time();
        print("{}", msg);
        fflush(stdout);
    }
    ~print_timer() {
        auto end = get_time();
        print(" in {:%H:%M:%S}\n", end - start);
    }

   private:
    string     msg;
    time_point start;
};
template <typename... Args>
inline print_timer print_timed(string_view format_str, const Args&... args) {
    return print_timer(format(format_str, args...));
}

// Logging to a sync
namespace impl {
inline auto log_callback = function<void(const string&)>{};
}

inline void set_log_callback(function<void(const string&)> callback) {
    impl::log_callback = callback;
}
template <typename... Args>
inline void log_msg(string_view format_str, const Args&... args) {
    if (impl::log_callback) impl::log_callback(format(format_str, args...));
}
template <typename... Args>
inline void log_info(string_view format_str, const Args&... args) {
    if (impl::log_callback)
        impl::log_callback(format(format_str, args...) + "\n");
}
template <typename... Args>
inline void log_error(string_view format_str, const Args&... args) {
    if (impl::log_callback)
        impl::log_callback(format(format_str, args...) + "\n");
}
struct log_timer {
    log_timer(const string& msg) : msg{msg} {
        start = get_time();
        log_msg("{}", msg);
    }
    ~log_timer() {
        auto end = get_time();
        log_msg("in {:%H:%M:%S}", end - start);
    }

   private:
    string     msg;
    time_point start;
};
template <typename... Args>
inline log_timer log_timed(string_view format_str, const Args&... args) {
    if (impl::log_callback) return log_timer(format(format_str, args...));
}

}  // namespace yocto

// Formatter for math types
namespace fmt {
// Formatter for math types
template <typename T, int N>
struct _formatter_base {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    constexpr auto format(const T& p, FormatContext& ctx) {
        if constexpr (N == 1) {
            return format_to(ctx.begin(), "[{}]", p[0]);
        } else if constexpr (N == 2) {
            return format_to(ctx.begin(), "[{}, {}]", p[0], p[1]);
        } else if constexpr (N == 3) {
            return format_to(ctx.begin(), "[{}, {}, {}]", p[0], p[1], p[2]);
        } else if constexpr (N == 4) {
            return format_to(
                ctx.begin(), "[{}, {}, {}, {}]", p[0], p[1], p[2], p[3]);
        } else {
            throw std::runtime_error("unsupported length");
        }
    }
};
// Formatter for math types
template <typename T, int N>
struct formatter<yocto::vec<T, N>> : _formatter_base<yocto::vec<T, N>, N> {};
template <typename T, int N, int M>
struct formatter<yocto::mat<T, N, M>>
    : _formatter_base<yocto::mat<T, N, M>, M> {};
template <typename T, int N>
struct formatter<yocto::affine<T, N>>
    : _formatter_base<yocto::affine<T, N>, N + 1> {};
template <typename T, int N>
struct formatter<yocto::frame<T, N>>
    : _formatter_base<yocto::frame<T, N>, N + 1> {};
template <typename T, int N>
struct formatter<yocto::bbox<T, N>> : _formatter_base<yocto::bbox<T, N>, 2> {};

}  // namespace fmt

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PYTHON-LIKE ITERATORS
// -----------------------------------------------------------------------------
namespace yocto {

// Range helpper (this should not be used directly)
template <typename T>
struct range_iterator {
    struct iterator {
        iterator& operator++() {
            pos++;
            return *this;
        }
        bool operator!=(const iterator& other) const {
            return pos != other.pos;
        }
        int operator*() const { return pos; }

       private:
        T pos = 0;
    };
    range_iterator(T start, T end) : first{first}, last{last} {}
    iterator begin() const { return {first}; }
    iterator end() const { return {last}; }

   private:
    T first = 0, last = 0;
};

// Python `range()` equivalent. Construct an object to iterate over a sequence.
template <typename T>
inline range_iterator<T> range(T max) {
    return range_iterator<T>{0, max};
}
template <typename T>
inline range_iterator<T> range(T min, T max) {
    return range_iterator<T>{min, max};
}

// Enumerate helper (this should not be used directly)
template <typename T>
struct enumerate_iterator {
    struct iterator {
        iterator(T* data, int pos) : data{data}, pos{pos} {}
        iterator& operator++() {
            pos++;
            return *this;
        }
        bool operator!=(const iterator& other) const {
            return pos != other.pos;
        }
        pair<int&, T&> operator*() const { return {pos, *(data + pos)}; }

       private:
        T*  data = nullptr;
        int pos  = 0;
    };
    enumerate_iterator(T* data, int size) : data{data}, size{size} {}
    iterator begin() const { return {data, 0}; }
    iterator end() const { return {data, size}; }

   private:
    T*  data = nullptr;
    int size = 0;
};

// Python `enumerate()` equivalent. Construct an object that iteraterates over a
// sequence of elements and numbers them.
template <typename T>
inline enumerate_iterator<const T> enumerate(const vector<T>& vals) {
    return enumerate_iterator<const T>{vals.data(), vals.size()};
};
template <typename T>
inline enumerate_iterator<T> enumerate(vector<T>& vals) {
    return enumerate_iterator<T>{vals.data(), vals.size()};
};

// Vector append and concatenation
template <typename T>
inline vector<T>& operator+=(vector<T>& a, const T& b) {
    a.push_back(b);
    return a;
}
template <typename T>
inline vector<T>& operator+=(vector<T>& a, const vector<T>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
template <typename T>
inline vector<T>& operator+=(vector<T>& a, const initializer_list<T>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
template <typename T>
inline vector<T> operator+(const vector<T>& a, const T& b) {
    auto c = a;
    return c += b;
}
template <typename T>
inline vector<T> operator+(const vector<T>& a, const vector<T>& b) {
    auto c = a;
    return c += b;
}
template <typename T>
inline vector<T> operator+(const vector<T>& a, const initializer_list<T>& b) {
    auto c = a;
    return c += b;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF STRING FORMAT UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Format duration string from nanoseconds
inline string format_duration(int64_t duration) {
    auto elapsed = duration / 1000000;  // milliseconds
    auto hours   = (int)(elapsed / 3600000);
    elapsed %= 3600000;
    auto mins = (int)(elapsed / 60000);
    elapsed %= 60000;
    auto secs  = (int)(elapsed / 1000);
    auto msecs = (int)(elapsed % 1000);
    char buffer[256];
    sprintf(buffer, "%02d:%02d:%02d.%03d", hours, mins, secs, msecs);
    return buffer;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PATH UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

inline string normalize_path(const string& filename_) {
    auto filename = filename_;
    for (auto& c : filename)
        if (c == '\\') c = '/';
    if (filename.size() > 1 && filename[0] == '/' && filename[1] == '/') {
        throw std::invalid_argument("absolute paths are not supported");
        return filename_;
    }
    if (filename.size() > 3 && filename[1] == ':' && filename[2] == '/' &&
        filename[3] == '/') {
        throw std::invalid_argument("absolute paths are not supported");
        return filename_;
    }
    auto pos = (size_t)0;
    while ((pos = filename.find("//")) != filename.npos)
        filename = filename.substr(0, pos) + filename.substr(pos + 1);
    return filename;
}

// Get directory name (including '/').
inline string get_dirname(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('/');
    if (pos == string::npos) return "";
    return filename.substr(0, pos + 1);
}

// Get extension (not including '.').
inline string get_extension(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('.');
    if (pos == string::npos) return "";
    return filename.substr(pos + 1);
}

// Get filename without directory.
inline string get_filename(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('/');
    if (pos == string::npos) return filename;
    return filename.substr(pos + 1);
}

// Get filename without directory and extension.
inline string get_basename(const string& filename) {
    return get_noextension(get_filename(filename));
}

// Get extension.
inline string get_noextension(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('.');
    if (pos == string::npos) return filename;
    return filename.substr(0, pos);
}

// Check if a file can be opened for reading.
bool exists_file(const string& filename) {
    auto f = fopen(filename.c_str(), "r");
    if (!f) return false;
    fclose(f);
    return true;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF FILE READING
// -----------------------------------------------------------------------------
namespace yocto {

// Load a text file
inline void load_text(const string& filename, string& str) {
    // https://stackoverflow.com/questions/174531/how-to-read-the-content-of-a-file-to-a-string-in-c
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) throw runtime_error("cannot open file " + filename);
    fseek(fs, 0, SEEK_END);
    auto length = ftell(fs);
    fseek(fs, 0, SEEK_SET);
    str.resize(length);
    if (fread(str.data(), 1, length, fs) != length) {
        fclose(fs);
        throw runtime_error("cannot read file " + filename);
    }
    fclose(fs);
}

// Save a text file
inline void save_text(const string& filename, const string& str) {
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) throw runtime_error("cannot open file " + filename);
    if (fprintf(fs, "%s", str.c_str()) < 0) {
        fclose(fs);
        throw runtime_error("cannot write file " + filename);
    }
    fclose(fs);
}

// Load a binary file
inline void load_binary(const string& filename, vector<byte>& data) {
    // https://stackoverflow.com/questions/174531/how-to-read-the-content-of-a-file-to-a-string-in-c
    auto fs = fopen(filename.c_str(), "rb");
    if (!fs) throw runtime_error("cannot open file " + filename);
    fseek(fs, 0, SEEK_END);
    auto length = ftell(fs);
    fseek(fs, 0, SEEK_SET);
    data.resize(length);
    if (fread(data.data(), 1, length, fs) != length) {
        fclose(fs);
        throw runtime_error("cannot read file " + filename);
    }
    fclose(fs);
}

// Save a binary file
inline void save_binary(const string& filename, const vector<byte>& data) {
    auto fs = fopen(filename.c_str(), "wb");
    if (!fs) throw runtime_error("cannot open file " + filename);
    if (fwrite(data.data(), 1, data.size(), fs) != data.size()) {
        fclose(fs);
        throw runtime_error("cannot write file " + filename);
    }
    fclose(fs);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR CONCURRENCY UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// a simple concurrent queue that locks at every call
template <typename T>
inline concurrent_queue<T>::concurrent_queue() {}
template <typename T>
inline concurrent_queue<T>::concurrent_queue(const concurrent_queue<T>& other) {
    if (!other._queue.empty())
        throw std::invalid_argument("cannot copy full queue");
    clear();
}
template <typename T>
inline concurrent_queue<T>& concurrent_queue<T>::operator=(
    const concurrent_queue<T>& other) {
    if (!other._queue.empty())
        throw std::invalid_argument("cannot copy full queue");
    clear();
}

template <typename T>
inline bool concurrent_queue<T>::empty() {
    lock_guard<mutex> lock(_mutex);
    return _queue.empty();
}
template <typename T>
inline void concurrent_queue<T>::clear() {
    lock_guard<mutex> lock(_mutex);
    _queue.clear();
}
template <typename T>
inline void concurrent_queue<T>::push(const T& value) {
    lock_guard<mutex> lock(_mutex);
    _queue.push_back(value);
}
template <typename T>
inline bool concurrent_queue<T>::try_pop(T& value) {
    lock_guard<mutex> lock(_mutex);
    if (_queue.empty()) return false;
    value = _queue.front();
    _queue.pop_front();
    return true;
}

// Runs a rask as an asycnrhonous operation.
template <typename Function>
inline auto async(Function&& function) {
    return std::async(std::launch::async, std::forward<Function>(function));
}
template <typename Function, typename... Args>
inline auto async(Function&& function, Args&&... args) {
    return std::async(std::launch::async, std::forward<Function>(function),
        std::forward<Args>(args)...);
}

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms.
template <typename Func>
inline void parallel_for(size_t begin, size_t end, const Func& func,
    atomic<bool>* cancel, bool serial) {
    if (serial) {
        for (auto idx = begin; idx < end; idx++) {
            if (cancel && *cancel) break;
            func(idx);
        }
    } else {
        auto           futures  = vector<future<void>>{};
        auto           nthreads = thread::hardware_concurrency();
        atomic<size_t> next_idx(begin);
        for (auto thread_id = 0; thread_id < nthreads; thread_id++) {
            futures.emplace_back(async([&func, &next_idx, cancel, end]() {
                while (true) {
                    if (cancel && *cancel) break;
                    auto idx = next_idx.fetch_add(1);
                    if (idx >= end) break;
                    func(idx);
                }
            }));
        }
        for (auto& f : futures) f.get();
    }
}

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes the integer index.
template <typename Func>
inline void parallel_for(
    size_t num, const Func& func, atomic<bool>* cancel, bool serial) {
    parallel_for(0, num, func, cancel, serial);
}

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes a reference to a `T`.
template <typename T, typename Func>
inline void parallel_foreach(
    vector<T>& values, const Func& func, atomic<bool>* cancel, bool serial) {
    parallel_for(
        0, (int)values.size(), [&func, &values](int idx) { func(values[idx]); },
        cancel, serial);
}
template <typename T, typename Func>
inline void parallel_foreach(const vector<T>& values, const Func& func,
    atomic<bool>* cancel, bool serial) {
    parallel_for(
        0, (int)values.size(), [&func, &values](int idx) { func(values[idx]); },
        cancel, serial);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// FILE UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// A file holder that closes a file when destructed. Useful for RIIA
struct file_holder {
    FILE* fs = nullptr;

    file_holder(const file_holder&) = delete;
    file_holder& operator=(const file_holder&) = delete;
    ~file_holder() {
        if (fs) fclose(fs);
    }
};

// Opens a file returing a handle with RIIA
inline file_holder open_input_file(
    const string& filename, bool binary = false) {
    auto fs = fopen(filename.c_str(), !binary ? "rt" : "rb");
    if (!fs) throw io_error("could not open file " + filename);
    return {fs};
}
inline file_holder open_output_file(
    const string& filename, bool binary = false) {
    auto fs = fopen(filename.c_str(), !binary ? "wt" : "wb");
    if (!fs) throw io_error("could not open file " + filename);
    return {fs};
}

// Read a line
inline bool read_line(FILE* fs, char* buffer, size_t size) {
    return fgets(buffer, size, fs) != nullptr;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF FAST PARSING
// -----------------------------------------------------------------------------
namespace yocto {

inline void skip_whitespace(string_view& str) {
    auto pos = str.find_first_not_of(" \t\r\n");
    if (pos == str.npos) {
        str.remove_prefix(str.size());
    } else {
        str.remove_prefix(pos);
    }
}
inline void trim_whitespace(string_view& str) {
    auto front = str.find_first_not_of(" \t\r\n");
    if (front == str.npos) {
        str.remove_prefix(str.size());
    } else {
        str.remove_prefix(front);
    }
    auto back = str.find_last_not_of(" \t\r\n");
    if (back == str.npos) {
        str.remove_suffix(str.size());
    } else {
        str.remove_suffix(str.size() - back - 1);
    }
}
inline void remove_comment_(string_view& str, char comment_char = '#') {
    auto pos = str.find(comment_char);
    if (pos == str.npos) return;
    str.remove_suffix(str.length() - pos);
}
inline void remove_comment_and_newline(
    string_view& str, char comment_char = '#') {
    str.remove_suffix(1);
    auto pos = str.find(comment_char);
    if (pos == str.npos) return;
    str.remove_suffix(str.length() - pos);
}

// Parse values from a string
inline void parse_value(string_view& str, int& value) {
    char* end = nullptr;
    value     = (int)strtol(str.data(), &end, 10);
    if (str == end) throw io_error("cannot parse value");
    str.remove_prefix(end - str.data());
}
inline void parse_value(string_view& str, bool& value) {
    auto valuei = 0;
    parse_value(str, valuei);
    value = (bool)valuei;
}
inline void parse_value(string_view& str, float& value) {
    char* end = nullptr;
    value     = strtof(str.data(), &end);
    if (str == end) throw io_error("cannot parse value");
    str.remove_prefix(end - str.data());
}
inline void parse_value(string_view& str, double& value) {
    char* end = nullptr;
    value     = strtod(str.data(), &end);
    if (str == end) throw io_error("cannot parse value");
    str.remove_prefix(end - str.data());
}
inline void parse_value(string_view& str, string& value, bool quoted = false) {
    skip_whitespace(str);
    if (str.empty()) throw io_error("cannot parse value");
    if (!quoted) {
        auto pos = str.find_first_of(" \t\r\n");
        if (pos == str.npos) {
            value = str;
            str.remove_prefix(str.length());
        } else {
            value = str.substr(0, pos);
            str.remove_prefix(pos);
        }
    } else {
        if (str.front() != '"') throw io_error("cannot parse value");
        str.remove_prefix(1);
        if (str.empty()) throw io_error("cannot parse value");
        auto pos = str.find('"');
        if (pos == str.npos) throw io_error("cannot parse value");
        value = str.substr(0, pos);
        str.remove_prefix(pos + 1);
    }
}
template <typename T, size_t N>
inline void parse_value(
    string_view& str, array<T, N>& value, bool in_brackets = false) {
    if (!in_brackets) {
        for (auto i = 0; i < N; i++) parse_value(str, value[i]);
    } else {
        skip_whitespace(str);
        if (str.empty() || str.front() != '[')
            throw io_error("cannot parse value");
        for (auto i = 0; i < N; i++) {
            if (i) {
                skip_whitespace(str);
                if (str.empty() || str.front() != ',')
                    throw io_error("cannot parse value");
            }
            parse_value(str, value[i]);
        }
        skip_whitespace(str);
        if (str.empty() || str.front() != ']')
            throw io_error("cannot parse value");
    }
}
template <typename T, int N>
inline void parse_value(
    string_view& str, vec<T, N>& value, bool in_brackets = false) {
    parse_value(str, (array<T, N>&)value, in_brackets);
}
template <typename T, int N>
inline void parse_value(
    string_view& str, frame<T, N>& value, bool in_brackets = false) {
    parse_value(str, (array<T, N*(N + 1)>&)value, in_brackets);
}
template <typename T, int N, int M>
inline void parse_value(
    string_view& str, mat<T, N, M>& value, bool in_brackets = false) {
    parse_value(str, (array<T, N * M>&)value, in_brackets);
}

template <typename T>
inline void parse_value_or_empty(string_view& str, T& value) {
    skip_whitespace(str);
    if (str.empty()) {
        value = T{};
    } else {
        parse_value(str, value);
    }
}

inline bool is_whitespace(const string_view& str) {
    return str.find_first_not_of(" \t\r\n") == str.npos;
}
inline bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
inline bool is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

inline void parse_varname(string_view& str, string& value) {
    skip_whitespace(str);
    if (str.empty()) throw io_error("cannot parse value");
    if (!is_alpha(str.front())) throw io_error("cannot parse value");
    auto pos = 0;
    while (is_alpha(str[pos]) || str[pos] == '_' || is_digit(str[pos])) {
        pos += 1;
        if (pos >= str.size()) break;
    }
    value = str.substr(0, pos);
    str.remove_prefix(pos);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// PYTHON-LIKE STRING OPERATIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Check if we start or end with a sequence
inline bool startswith(string_view str, string_view substr) {
    return str.find(substr) == 0;
}
inline bool endswith(string_view str, string_view substr) {
    return str.rfind(substr) == str.size() - substr.size();
}
inline void split_view(string_view str, vector<string_view>& splits,
    string_view delimiters = " \t\r\n", bool trim_empty = true) {
    splits.clear();
    while (!str.empty()) {
        auto pos = str.find_first_of(delimiters);
        if (pos == string_view::npos) {
            splits.push_back(str);
            break;
        } else if (pos == 0) {
            if (!trim_empty) splits.push_back(str.substr(0, 1));
            str.remove_prefix(1);
        } else {
            splits.push_back(str.substr(0, pos));
            str.remove_prefix(pos + 1);
        }
    }
}
inline vector<string_view> split_view(string_view str,
    string_view delimiters = " \t\r\n", bool trim_empty = true) {
    auto splits = vector<string_view>{};
    split_view(str, splits, delimiters, trim_empty);
    return splits;
}
inline vector<string> split(const string& str,
    string_view delimiters = " \t\r\n", bool trim_empty = true) {
    auto splits = vector<string_view>{};
    split_view(str, splits, delimiters, trim_empty);
    auto splits_str = vector<string>();
    for (auto split : splits) splits_str.push_back(string(split));
    return splits_str;
}
inline void splitlines(string_view str, vector<string_view>& splits) {
    splits.clear();
    while (!str.empty()) {
        auto pos = min(str.find("\n"), str.find("\r\n"));
        if (pos == string_view::npos) {
            splits.push_back(str);
            break;
        } else {
            splits.push_back(str.substr(0, pos));
            str.remove_prefix(pos + (str.front() == '\n' ? 0 : 1));
        }
    }
}
inline vector<string_view> splitlines(string_view str) {
    auto splits = vector<string_view>{};
    splitlines(str, splits);
    return splits;
}
inline vector<string> splitlines(const string& str) {
    auto splits = vector<string_view>{};
    splitlines(str, splits);
    auto splits_str = vector<string>();
    for (auto split : splits) splits_str.push_back(string(split));
    return splits_str;
}

inline string replace(string_view str, string_view from, string_view to) {
    // https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
    auto replaced = ""s;
    while (!str.empty()) {
        auto pos = str.find(from);
        if (pos == string_view::npos) {
            replaced += str;
            break;
        } else {
            replaced += str.substr(0, pos);
            replaced += to;
            str.remove_prefix(pos + from.size());
        }
    }
    return replaced;
}

}  // namespace yocto

#endif
