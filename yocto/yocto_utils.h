//
// # Yocto/Utils: Tiny collection of utilities to support Yocto/GL
//
//
// Yocto/Utils is a collection of utilities used in writing other Yocto/GL
// libraries and example applications. We support printing and parsing builting
// and Yocto/Math values, parsing command line arguments, simple path
// manipulation, file lading/saving and basic concurrency utilities.
//
//
// ## Printing and parsing values
//
// Use `format()` to format a string using `{}` as placeholder and `print()`
// to print it. Use `parse()` to parse a value from a string.
//
//
// ## Command-Line Parsing
//
// We provide a simple, immediate-mode, command-line parser. The parser
// works in an immediate-mode manner since it reads each value as you call each
// function, rather than building a data structure and parsing offline. We
// support option and position arguments, automatic help generation, and
// error checking.
//
// 1. initialize the parser with `make_cmdline_parser(argc, argv, help)`
// 2. read a value with `value = parse_argument(parser, name, default, help)`
//    - is name starts with '--' or '-' then it is an option
//    - otherwise it is a positional arguments
//    - options and arguments may be intermixed
//    - the type of each option is determined by the default value `default`
//    - the value is parsed on the stop
// 3. finished parsing with `check_cmdline(parser)`
//    - if an error occurred, the parser will exit and print a usage message
//
//
// ## Path manipulation
//
// We define a few path manipulation utilities to split and join path components.
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
// Copyright (c) 2016 -- 2018 Fabio Pellacini
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

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

using std::atomic;
using std::deque;
using std::lock_guard;
using std::mutex;
using std::string;
using std::thread;
using std::vector;
using namespace std::string_literals;
using namespace std::chrono_literals;

}  // namespace yocto

// -----------------------------------------------------------------------------
// PRINT/PARSE UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Formats a string `fmt` with values taken from `args`. Uses `{}` as
// placeholder.
template <typename... Args>
inline string format(const string& fmt, const Args&... args);

// Converts to string.
template <typename T>
inline string to_string(const T& value);

// Prints a formatted string to stdout or file.
template <typename... Args>
inline bool print(FILE* fs, const string& fmt, const Args&... args);
template <typename... Args>
inline bool print(const string& fmt, const Args&... args) {
    return print(stdout, fmt, args...);
}

// Format duration string from nanoseconds
inline string format_duration(int64_t duration);
// Format a large integer number in human readable form
inline string format_num(uint64_t num);

// Parse a list of space separated values.
template <typename... Args>
inline bool parse(const string& str, Args&... args);

// get time in nanoseconds - useful only to compute difference of times
inline int64_t get_time() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// LOGGING UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Log info/error/fatal/trace message
template <typename... Args>
inline void log_info(const string& fmt, const Args&... args);
template <typename... Args>
inline void log_error(const string& fmt, const Args&... args);
template <typename... Args>
inline void log_warning(const string& fmt, const Args&... args);
template <typename... Args>
inline void log_fatal(const string& fmt, const Args&... args);

// Setup logging
inline void set_log_console(bool enabled);
inline void set_log_file(const string& filename, bool append = false);

// Log traces for timing and program debugging
struct log_scope;
template <typename... Args>
inline void log_trace(const string& fmt, const Args&... args);
template <typename... Args>
inline log_scope log_trace_begin(const string& fmt, const Args&... args);
template <typename... Args>
inline void log_trace_end(log_scope& scope);
template <typename... Args>
inline log_scope log_trace_scoped(const string& fmt, const Args&... args);

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMMEDIATE-MODE COMMAND LINE PARSING
// -----------------------------------------------------------------------------
namespace yocto {

// Command line parser data. All data should be considered private.
struct cmdline_parser {
    vector<string> args      = {};  // command line arguments
    string         usage_cmd = "";  // program name
    string         usage_hlp = "";  // program help
    string         usage_opt = "";  // options help
    string         usage_arg = "";  // arguments help
    string         error     = "";  // current parse error
};

// Initialize a command line parser.
inline cmdline_parser make_cmdline_parser(
    int argc, char** argv, const string& usage, const string& cmd = "");
// check if any error occurred and exit appropriately
inline void check_cmdline(cmdline_parser& parser);

// Parse an int, float, string, vecXX and bool option or positional argument.
// Options's names starts with "--" or "-", otherwise they are arguments.
// vecXX options use space-separated values but all in one argument
// (use " or ' from the common line). Booleans are flags.
template <typename T>
inline T parse_argument(cmdline_parser& parser, const string& name, T def,
    const string& usage, bool req = false);
// Parse all arguments left on the command line.
template <typename T>
inline vector<T> parse_arguments(cmdline_parser& parser, const string& name,
    const vector<T>& def, const string& usage, bool req = false);
// Parse a labeled enum, with enum values that are successive integers.
template <typename T>
inline T parse_argument(cmdline_parser& parser, const string& name, T def,
    const string& usage, const vector<string>& labels, bool req = false);

// Parse an int, float, string, vecXX and bool option or positional argument.
// Options's names starts with "--" or "-", otherwise they are arguments.
// vecXX options use space-separated values but all in one argument
// (use " or ' from the common line). Booleans are flags.
template <typename T>
inline bool parse_argument_ref(cmdline_parser& parser, const string& name,
    T& val, const string& usage, bool req = false);
// Parse all arguments left on the command line.
template <typename T>
inline bool parse_arguments_ref(cmdline_parser& parser, const string& name,
    vector<T>& val, const string& usage, bool req = false);
// Parse a labeled enum, with enum values that are successive integers.
template <typename T>
inline bool parse_argument_ref(cmdline_parser& parser, const string& name,
    T& val, const string& usage, const vector<string>& labels, bool req = false);

}  // namespace yocto

// -----------------------------------------------------------------------------
// PATH UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Normalize path delimiters.
inline string normalize_path(const string& filename);
// Get directory name (not including '/').
inline string get_dirname(const string& filename);
// Get extension (not including '.').
inline string get_extension(const string& filename);
// Get filename without directory.
inline string get_filename(const string& filename);
// Replace extension.
inline string replace_extension(const string& filename, const string& ext);

// Check if a file can be opened for reading.
inline bool exists_file(const string& filename);

}  // namespace yocto

// -----------------------------------------------------------------------------
// FILE IO
// -----------------------------------------------------------------------------
namespace yocto {

// Load/save a text file
inline bool load_text(const string& filename, string& str);
inline bool save_text(const string& filename, const string& str);

// Load/save a binary file
inline bool load_binary(const string& filename, vector<byte>& data);
inline bool save_binary(const string& filename, const vector<byte>& data);

}  // namespace yocto

// -----------------------------------------------------------------------------
// CONCURRENCY UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// a simple concurrent queue that locks at every call
template <typename T>
struct concurrent_queue {
    concurrent_queue() {}
    concurrent_queue(const concurrent_queue& other) {
        if (!other._queue.empty()) log_error("cannot copy full queue");
        clear();
    }
    concurrent_queue& operator=(const concurrent_queue& other) {
        if (!other._queue.empty()) log_error("cannot copy full queue");
        clear();
    }

    bool empty() {
        lock_guard<mutex> lock(_mutex);
        return _queue.empty();
    }
    void clear() {
        lock_guard<mutex> lock(_mutex);
        _queue.clear();
    }
    void push(const T& value) {
        lock_guard<mutex> lock(_mutex);
        _queue.push_back(value);
    }
    bool try_pop(T& value) {
        lock_guard<mutex> lock(_mutex);
        if (_queue.empty()) return false;
        value = _queue.front();
        _queue.pop_front();
        return true;
    }

    mutex    _mutex;
    deque<T> _queue;
};

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes the integer index.
template <typename Func>
inline void parallel_for(int begin, int end, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false);
template <typename Func>
inline void parallel_for(int num, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false) {
    parallel_for(0, num, func, cancel, serial);
}

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes a reference to a `T`.
template <typename T, typename Func>
inline void parallel_foreach(vector<T>& values, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false) {
    parallel_for(0, (int)values.size(),
        [&func, &values](int idx) { func(values[idx]); }, cancel, serial);
}
template <typename T, typename Func>
inline void parallel_foreach(const vector<T>& values, const Func& func,
    atomic<bool>* cancel = nullptr, bool serial = false) {
    parallel_for(0, (int)values.size(),
        [&func, &values](int idx) { func(values[idx]); }, cancel, serial);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF STRING/TIME UTILITIES FOR CLI APPLICATIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Prints basic types
inline bool print_value(string& str, const string& value) {
    str += value;
    return true;
}
inline bool print_value(string& str, const char* value) {
    str += value;
    return true;
}
inline bool print_value(string& str, int value) {
    str += std::to_string(value);
    return true;
}
inline bool print_value(string& str, float value) {
    str += std::to_string(value);
    return true;
}
inline bool print_value(string& str, double value) {
    str += std::to_string(value);
    return true;
}
template <typename T>
inline bool print_value(string& str, const T* value) {
    char buffer[512];
    sprintf(buffer, "%p", value);
    str += buffer;
    return true;
}

template <typename T, size_t N>
inline bool print_value(string& str, const array<T, N>& value) {
    for (auto i = 0; i < N; i++) {
        if (i) str += " ";
        str += std::to_string(value[i]);
    }
    return true;
}
template <typename T>
inline bool print_values(string& str, const T* values, int N) {
    for (auto i = 0; i < N; i++) {
        if (i) str += " ";
        str += std::to_string(values[i]);
    }
    return true;
}

// Print compound types.
template <typename T, int N>
inline bool print_value(string& str, const vec<T, N>& v) {
    return print_values(str, &v[0], N);
}
template <typename T, int N, int M>
inline bool print_value(string& str, const mat<T, N, M>& v) {
    return print_values(str, &v[0][0], N * M);
}
template <typename T, int N>
inline bool print_value(string& str, const frame<T, N>& v) {
    return print_values(str, &v[0][0], N * (N + 1));
}
template <typename T, int N>
inline bool print_value(string& str, const bbox<T, N>& v) {
    return print_values(str, &v[0][0], N * 2);
}
template <typename T, int N>
inline bool print_value(string& str, const ray<T, N>& v) {
    return print_values(str, &v[0][0], N * 2 + 2);
}

// Prints a string.
inline bool print_next(string& str, const string& fmt) {
    return print_value(str, fmt);
}
template <typename Arg, typename... Args>
inline bool print_next(
    string& str, const string& fmt, const Arg& arg, const Args&... args) {
    auto pos = fmt.find("{}");
    if (pos == string::npos) return print_value(str, fmt);
    if (!print_value(str, fmt.substr(0, pos))) return false;
    if (!print_value(str, arg)) return false;
    return print_next(str, fmt.substr(pos + 2), args...);
}

// Formats a string `fmt` with values taken from `args`. Uses `{}` as
// placeholder.
template <typename... Args>
inline string format(const string& fmt, const Args&... args) {
    auto str = string();
    print_next(str, fmt, args...);
    return str;
}

// Prints a string.
template <typename... Args>
inline bool print(FILE* fs, const string& fmt, const Args&... args) {
    auto str = format(fmt, args...);
    return fprintf(fs, "%s", str.c_str()) >= 0;
}

// Converts to string.
template <typename T>
inline string to_string(const T& value) {
    auto str = string();
    print_value(str, value);
    return str;
}

// Trivial wrapper used for simplicity
struct parse_string_view {
    const char* str = nullptr;
};

// Prints basic types to string
inline bool parse_value(parse_string_view& str, string& value) {
    while (*str.str && std::isspace((unsigned char)*str.str)) str.str++;
    if (!*str.str) return false;
    auto pos = 0;
    char buffer[4096];
    while (*str.str && !std::isspace((unsigned char)*str.str) &&
           pos < sizeof(buffer)) {
        buffer[pos] = *str.str;
        str.str++;
        pos++;
    }
    if (pos >= sizeof(buffer)) return false;
    buffer[pos] = 0;
    value       = buffer;
    return true;
}
inline bool parse_value(parse_string_view& str, int& value) {
    char* end = nullptr;
    value     = (int)strtol(str.str, &end, 10);
    if (str.str == end) return false;
    str.str = end;
    // auto n = 0;
    // if (sscanf(str.str, "%d%n", &value, &n) != 1) return false;
    // str.str += n;
    return true;
}
inline bool parse_value(parse_string_view& str, float& value) {
    char* end = nullptr;
    value     = strtof(str.str, &end);
    if (str.str == end) return false;
    str.str = end;
    // auto n = 0;
    // if (sscanf(str.str, "%f%n", &value, &n) != 1) return false;
    // str.str += n;
    return true;
}
inline bool parse_value(parse_string_view& str, double& value) {
    char* end = nullptr;
    value     = strtod(str.str, &end);
    if (str.str == end) return false;
    str.str = end;
    // auto n = 0;
    // if (sscanf(str.str, "%lf%n", &value, &n) != 1) return false;
    // str.str += n;
    return true;
}
inline bool parse_value(parse_string_view& str, bool& value) {
    auto ivalue = 0;
    if (!parse_value(str, ivalue)) return false;
    value = (bool)ivalue;
    return true;
}

// Print compound types
template <typename T, size_t N>
inline bool parse_value(parse_string_view& str, array<T, N>& value) {
    for (auto i = 0; i < N; i++) {
        if (!parse_value(str, value[i])) return false;
    }
    return true;
}
template <typename T>
inline bool parse_values(parse_string_view& str, T* values, int N) {
    for (auto i = 0; i < N; i++) {
        if (!parse_value(str, values[i])) return false;
    }
    return true;
}

// Data acess
template <typename T, int N>
inline bool parse_value(parse_string_view& str, vec<T, N>& v) {
    return parse_values(str, &v[0], N);
}
template <typename T, int N, int M>
inline bool parse_value(parse_string_view& str, mat<T, N, M>& v) {
    return parse_values(str, &v[0][0], N * M);
}
template <typename T, int N>
inline bool parse_value(parse_string_view& str, frame<T, N>& v) {
    return parse_values(str, &v[0][0], N * (N + 1));
}
template <typename T, int N>
inline bool parse_value(parse_string_view& str, bbox<T, N>& v) {
    return parse_values(str, &v[0][0], N * 2);
}
template <typename T, int N>
inline bool parse_value(parse_string_view& str, ray<T, N>& v) {
    return parse_values(str, &v.origin[0], N * 2 + 2);
}

// Prints a string.
inline bool parse_next(parse_string_view& str) { return true; }
template <typename Arg, typename... Args>
inline bool parse_next(parse_string_view& str, Arg& arg, Args&... args) {
    if (!parse_value(str, arg)) return false;
    return parse_next(str, args...);
}

// Returns trus if this is white space
inline bool is_whitespace(parse_string_view str) {
    while (*str.str && isspace((unsigned char)*str.str)) str.str++;
    return *str.str == 0;
}

// Parse a list of space separated values.
template <typename... Args>
inline bool parse(const string& str, Args&... args) {
    auto view = parse_string_view{str.c_str()};
    if (!parse_next(view, args...)) return false;
    return is_whitespace(view);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF LOGGING UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Logging configutation
inline bool& _log_console() {
    static auto _log_console = true;
    return _log_console;
}
inline FILE*& _log_filestream() {
    static auto _log_filestream = (FILE*)nullptr;
    return _log_filestream;
}

// Logs a message
inline void log_message(const char* lbl, const char* msg) {
    if (_log_console()) {
        printf("%s\n", msg);
        fflush(stdout);
    }
    if (_log_filestream()) {
        fprintf(_log_filestream(), "%s %s\n", lbl, msg);
        fflush(_log_filestream());
    }
}

// Log info/error/fatal/trace message
template <typename... Args>
inline void log_info(const string& fmt, const Args&... args) {
    log_message("INFO ", format(fmt, args...).c_str());
}
template <typename... Args>
inline void log_error(const string& fmt, const Args&... args) {
    log_message("ERROR", format(fmt, args...).c_str());
}
template <typename... Args>
inline void log_warning(const string& fmt, const Args&... args) {
    log_message("WARN ", format(fmt, args...).c_str());
}
template <typename... Args>
inline void log_fatal(const string& fmt, const Args&... args) {
    log_message("FATAL", format(fmt, args...).c_str());
    exit(1);
}

// Log traces for timing and program debugging
struct log_scope {
    string  message    = "";
    int64_t start_time = -1;
    bool    scoped     = false;
    ~log_scope();
};
template <typename... Args>
inline void log_trace(const string& fmt, const Args&... args) {
    log_message("TRACE", format(fmt, args...).c_str());
}
template <typename... Args>
inline log_scope log_trace_begin(const string& fmt, const Args&... args) {
    auto message = format(fmt, args...);
    log_trace(message + " [started]");
    return {message, get_time(), false};
}
template <typename... Args>
inline void log_trace_end(log_scope& scope) {
    if (scope.start_time >= 0) {
        log_trace(scope.message + " [ended: " +
                  format_duration(get_time() - scope.start_time) + "]");
    } else {
        log_trace(scope.message + " [ended]");
    }
}
template <typename... Args>
inline log_scope log_trace_scoped(const string& fmt, const Args&... args) {
    auto message = format(fmt, args...);
    log_trace(message + " [started]");
    return {message, get_time(), true};
}
inline log_scope::~log_scope() {
    if (scoped) log_trace_end(*this);
}

// Configure the logging
inline void set_log_console(bool enabled) { _log_console() = enabled; }
inline void set_log_file(const string& filename, bool append) {
    if (_log_filestream()) {
        fclose(_log_filestream());
        _log_filestream() = nullptr;
    }
    if (filename.empty()) return;
    _log_filestream() = fopen(filename.c_str(), append ? "at" : "wt");
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
// Format a large integer number in human readable form
inline string format_num(uint64_t num) {
    auto rem = num % 1000;
    auto div = num / 1000;
    if (div > 0) return format_num(div) + "," + std::to_string(rem);
    return std::to_string(rem);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF COMMAND-LINE PARSING
// -----------------------------------------------------------------------------
namespace yocto {

// initialize a command line parser
inline cmdline_parser make_cmdline_parser(
    int argc, char** argv, const string& usage, const string& cmd) {
    auto parser      = cmdline_parser{};
    parser.args      = {argv + 1, argv + argc};
    parser.usage_cmd = (cmd.empty()) ? argv[0] : cmd;
    parser.usage_hlp = usage;
    return parser;
}

// check if option or argument
inline bool is_option(const string& name) {
    return name.size() > 1 && name.front() == '-';
}

// get names from string
inline vector<string> get_option_names(const string& name_) {
    auto names = vector<string>();
    auto name  = name_;
    while (name.find(',') != name.npos) {
        names.push_back(name.substr(0, name.find(',')));
        name = name.substr(name.find(',') + 1);
    }
    names.push_back(name);
    return names;
}

// add help
inline string get_option_usage(const string& name, const string& usage,
    const string& def_, const vector<string>& choices) {
    auto def = def_;
    if (def != "") def = "[" + def + "]";
    auto namevar = name;
    if (name != "") namevar += " " + name;
    char buffer[4096];
    sprintf(
        buffer, "  %-24s %s %s\n", namevar.c_str(), usage.c_str(), def.c_str());
    auto usagelines = string(buffer);
    if (!choices.empty()) {
        usagelines += "        accepted values:";
        for (auto& c : choices) usagelines += " " + c;
        usagelines += "\n";
    }
    return usagelines;
}

// print cmdline help
inline void print_cmdline_usage(const cmdline_parser& parser) {
    printf("%s: %s\n", parser.usage_cmd.c_str(), parser.usage_hlp.c_str());
    printf("usage: %s %s %s\n\n", parser.usage_cmd.c_str(),
        (parser.usage_opt.empty()) ? "" : "[options]",
        (parser.usage_arg.empty()) ? "" : "arguments");
    if (!parser.usage_opt.empty()) {
        printf("options:\n");
        printf("%s\n", parser.usage_opt.c_str());
    }
    if (!parser.usage_arg.empty()) {
        printf("arguments:\n");
        printf("%s\n", parser.usage_arg.c_str());
    }
}

// Parse a flag. Name should start with either "--" or "-".
inline bool parse_flag_argument(cmdline_parser& parser, const string& name,
    bool& value, const string& usage);

// check if any error occurred and exit appropriately
inline void check_cmdline(cmdline_parser& parser) {
    auto help = false;
    if (parse_flag_argument(parser, "--help,-?", help, "print help")) {
        print_cmdline_usage(parser);
        exit(0);
    }
    if (!parser.args.empty()) parser.error += "unmatched arguments remaining\n";
    if (!parser.error.empty()) {
        printf("error: %s", parser.error.c_str());
        print_cmdline_usage(parser);
        exit(1);
    }
}

// Parse an option string. Name should start with "--" or "-".
template <typename T>
inline bool parse_option_argument(cmdline_parser& parser, const string& name,
    T& value, const string& usage, bool req, const vector<string>& choices) {
    parser.usage_opt += get_option_usage(name, usage, to_string(value), choices);
    if (parser.error != "") return false;
    auto names = get_option_names(name);
    auto pos   = parser.args.end();
    for (auto& name : names) {
        pos = std::min(
            pos, std::find(parser.args.begin(), parser.args.end(), name));
    }
    if (pos == parser.args.end()) {
        if (req) parser.error += "missing value for " + name;
        return false;
    }
    if (pos == parser.args.end() - 1) {
        parser.error += "missing value for " + name;
        return false;
    }
    auto vals = *(pos + 1);
    parser.args.erase(pos, pos + 2);
    if (!choices.empty() &&
        std::find(choices.begin(), choices.end(), vals) == choices.end()) {
        parser.error += "bad value for " + name;
        return false;
    }
    auto new_value = value;
    if (!parse(vals, new_value)) {
        parser.error += "bad value for " + name;
        return false;
    }
    value = new_value;
    return true;
}

// Parse an argument string. Name should not start with "--" or "-".
template <typename T>
inline bool parse_positional_argument(cmdline_parser& parser, const string& name,
    T& value, const string& usage, bool req, const vector<string>& choices) {
    parser.usage_arg += get_option_usage(name, usage, to_string(value), choices);
    if (parser.error != "") return false;
    auto pos = std::find_if(parser.args.begin(), parser.args.end(),
        [](auto& v) { return v[0] != '-'; });
    if (pos == parser.args.end()) {
        if (req) parser.error += "missing value for " + name;
        return false;
    }
    auto vals = *pos;
    parser.args.erase(pos);
    if (!choices.empty() &&
        std::find(choices.begin(), choices.end(), vals) == choices.end()) {
        parser.error += "bad value for " + name;
        return false;
    }
    auto new_value = value;
    if (!parse(vals, new_value)) {
        parser.error += "bad value for " + name;
        return false;
    }
    value = new_value;
    return true;
}

// Parse all left argument strings. Name should not start with "--" or "-".
template <typename T>
inline bool parse_positional_arguments(cmdline_parser& parser,
    const string& name, vector<T>& values, const string& usage, bool req) {
    auto defs = string();
    for (auto& d : values) defs += " " + d;
    parser.usage_arg += get_option_usage(name, usage, defs, {});
    if (parser.error != "") return false;
    auto pos = std::find_if(parser.args.begin(), parser.args.end(),
        [](auto& v) { return v[0] != '-'; });
    if (pos == parser.args.end()) {
        if (req) parser.error += "missing value for " + name;
        return false;
    }
    auto vals = vector<string>{pos, parser.args.end()};
    parser.args.erase(pos, parser.args.end());
    auto new_values = values;
    new_values.resize(vals.size());
    for (auto i = 0; i < vals.size(); i++) {
        if (!parse(vals[i], new_values[i])) {
            parser.error += "bad value for " + name;
            return false;
        }
    }
    values = new_values;
    return true;
}

// Parse a flag. Name should start with either "--" or "-".
inline bool parse_flag_argument(cmdline_parser& parser, const string& name,
    bool& value, const string& usage) {
    parser.usage_opt += get_option_usage(name, usage, "", {});
    if (parser.error != "") return false;
    auto names = get_option_names(name);
    auto pos   = parser.args.end();
    for (auto& name : names)
        pos = std::min(
            pos, std::find(parser.args.begin(), parser.args.end(), name));
    if (pos == parser.args.end()) return false;
    parser.args.erase(pos);
    value = !value;
    return true;
}

// Parse an integer, float, string. If name starts with "--" or "-", then it is
// an option, otherwise it is a position argument.
template <typename T>
inline bool parse_argument_ref(cmdline_parser& parser, const string& name,
    T& value, const string& usage, bool req) {
    return is_option(name) ?
               parse_option_argument(parser, name, value, usage, req, {}) :
               parse_positional_argument(parser, name, value, usage, req, {});
}
template <>
inline bool parse_argument_ref<bool>(cmdline_parser& parser, const string& name,
    bool& value, const string& usage, bool req) {
    return parse_flag_argument(parser, name, value, usage);
}

template <typename T>
inline bool parse_argument_ref(cmdline_parser& parser, const string& name,
    T& value, const string& usage, const vector<string>& labels, bool req) {
    auto values = labels.at((int)value);
    auto parsed = is_option(name) ? parse_option_argument(parser, name, values,
                                        usage, req, labels) :
                                    parse_positional_argument(parser, name,
                                        values, usage, req, labels);
    if (!parsed) return false;
    auto pos = std::find(labels.begin(), labels.end(), values);
    if (pos == labels.end()) return false;
    value = (T)(pos - labels.begin());
    return true;
}

// Parser an argument
template <typename T>
inline bool parse_arguments_ref(cmdline_parser& parser, const string& name,
    vector<T>& values, const string& usage, bool req) {
    return parse_positional_arguments(parser, name, values, usage, req);
}

// Parse an integer, float, string. If name starts with "--" or "-", then it is
// an option, otherwise it is a position argument.
template <typename T>
inline T parse_argument(cmdline_parser& parser, const string& name, T def,
    const string& usage, bool req) {
    auto value = def;
    if (!parse_argument_ref(parser, name, value, usage, req)) return def;
    return value;
}
template <>
inline bool parse_argument<bool>(cmdline_parser& parser, const string& name,
    bool def, const string& usage, bool req) {
    auto value = def;
    if (!parse_flag_argument(parser, name, value, usage)) return def;
    return value;
}

template <typename T>
inline T parse_argument(cmdline_parser& parser, const string& name, T def,
    const string& usage, const vector<string>& labels, bool req) {
    auto value = def;
    if (!parse_argument_ref(parser, name, value, usage, labels, req))
        return def;
    return value;
}

// Parser an argument
template <typename T>
inline vector<T> parse_arguments(cmdline_parser& parser, const string& name,
    const vector<T>& def, const string& usage, bool req) {
    auto values = vector<T>{};
    if (!parse_arguments_ref(parser, name, values, usage, req)) return def;
    return values;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PATH UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

string normalize_path(const string& filename_) {
    auto filename = filename_;
    for (auto& c : filename)
        if (c == '\\') c = '/';
    if (filename.size() > 1 && filename[0] == '/' && filename[1] == '/') {
        log_error("absolute paths are not supported");
        return filename_;
    }
    if (filename.size() > 3 && filename[1] == ':' && filename[2] == '/' &&
        filename[3] == '/') {
        log_error("absolute paths are not supported");
        return filename_;
    }
    auto pos = (size_t)0;
    while ((pos = filename.find("//")) != filename.npos)
        filename = filename.substr(0, pos) + filename.substr(pos + 1);
    return filename;
}

// Get directory name (not including '/').
string get_dirname(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('/');
    if (pos == string::npos) return "";
    return filename.substr(0, pos);
}

// Get extension (not including '.').
string get_extension(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('.');
    if (pos == string::npos) return "";
    return filename.substr(pos + 1);
}

// Get filename without directory.
string get_filename(const string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('/');
    if (pos == string::npos) return "";
    return filename.substr(pos + 1);
}

// Replace extension.
string replace_extension(const string& filename_, const string& ext_) {
    auto filename = normalize_path(filename_);
    auto ext      = normalize_path(ext_);
    if (ext.at(0) == '.') ext = ext.substr(1);
    auto pos = filename.rfind('.');
    if (pos == string::npos) return filename;
    return filename.substr(0, pos) + "." + ext;
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

// log io error
template <typename... Args>
inline void log_io_error(const string& fmt, const Args&... args) {
    log_error(fmt, args...);
}

// File stream wrapper
struct file_stream {
    string filename = "";
    string mode     = "";
    FILE*  fs       = nullptr;

    file_stream()                   = default;
    file_stream(const file_stream&) = delete;
    file_stream& operator=(const file_stream&) = delete;
    file_stream(file_stream&&)                 = default;
    file_stream& operator=(file_stream&&) = default;

    ~file_stream() {
        if (fs) {
            fclose(fs);
            fs = nullptr;
        }
    }

    operator bool() const { return fs; }
};

// Opens a file
inline file_stream open(const string& filename, const string& mode) {
    auto fs = fopen(filename.c_str(), mode.c_str());
    if (!fs) {
        log_io_error("cannot open {}", filename);
        return {};
    }
    return {filename, mode, fs};
}

// Close a file
inline bool close(file_stream& fs) {
    if (!fs) {
        log_io_error("cannot close {}", fs.filename);
        return false;
    }
    fclose(fs.fs);
    fs.fs = nullptr;
    return true;
}

// Gets the length of a file
inline size_t get_length(file_stream& fs) {
    if (!fs) return 0;
    fseek(fs.fs, 0, SEEK_END);
    auto fsize = ftell(fs.fs);
    fseek(fs.fs, 0, SEEK_SET);
    return fsize;
}

// Print to file
inline bool write_text(file_stream& fs, const string& str) {
    if (!fs) return false;
    if (fprintf(fs.fs, "%s", str.c_str()) < 0) {
        log_io_error("cannot write to {}", fs.filename);
        return false;
    }
    return true;
}

// Write to file
template <typename T>
inline bool write_value(file_stream& fs, const T& value) {
    if (!fs) return false;
    if (fwrite(&value, sizeof(T), 1, fs.fs) != 1) {
        log_io_error("cannot write to {}", fs.filename);
        return false;
    }
    return true;
}

// Write to file
template <typename T>
inline bool write_values(file_stream& fs, const vector<T>& vals) {
    if (!fs) return false;
    if (fwrite(vals.data(), sizeof(T), vals.size(), fs.fs) != vals.size()) {
        log_io_error("cannot write to {}", fs.filename);
        return false;
    }
    return true;
}

// Write to file
template <typename T>
inline bool write_values(file_stream& fs, size_t num, const T* vals) {
    if (!fs) return false;
    if (fwrite(vals, sizeof(T), num, fs.fs) != num) {
        log_io_error("cannot write to {}", fs.filename);
        return false;
    }
    return true;
}

// Print shortcut
template <typename... Args>
inline bool print(file_stream& fs, const string& fmt, const Args&... args) {
    if (!fs) return false;
    return write_text(fs, format(fmt, args...));
}

// Read binary data to fill the whole buffer
inline bool read_line(file_stream& fs, string& value) {
    if (!fs) return false;
    // TODO: make lkne as large as possible
    value = "";
    char buffer[4096];
    if (!fgets(buffer, 4096, fs.fs)) return false;
    value = string(buffer);
    return true;
}

// Read binary data to fill the whole buffer
template <typename T>
inline bool read_value(file_stream& fs, T& value) {
    if (!fs) return false;
    if (fread(&value, sizeof(T), 1, fs.fs) != 1) {
        log_io_error("cannot read from {}", fs.filename);
        return false;
    }
    return true;
}

// Read binary data to fill the whole buffer
template <typename T>
inline bool read_values(file_stream& fs, vector<T>& vals) {
    if (!fs) return false;
    if (fread(vals.data(), sizeof(T), vals.size(), fs.fs) != vals.size()) {
        log_io_error("cannot read from {}", fs.filename);
        return false;
    }
    return true;
}

// Read binary data to fill the whole buffer
template <typename T>
inline bool read_values(file_stream& fs, size_t num, T* vals) {
    if (!fs) return false;
    if (fread(vals, sizeof(T), num, fs.fs) != num) {
        log_io_error("cannot read from {}", fs.filename);
        return false;
    }
    return true;
}

// Load a text file
inline bool load_text(const string& filename, string& str) {
    // https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
    auto fs = open(filename, "rb");
    if (!fs) return false;
    auto buffer = vector<char>(get_length(fs));
    if (!read_values(fs, buffer)) return false;
    str = string{buffer.begin(), buffer.end()};
    return true;
}

// Save a text file
inline bool save_text(const string& filename, const string& str) {
    auto fs = open(filename, "wt");
    if (!fs) return false;
    if (!write_text(fs, str)) return false;
    return true;
}

// Load a binary file
inline bool load_binary(const string& filename, vector<byte>& data) {
    // https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
    auto fs = open(filename, "rb");
    if (!fs) return false;
    data = vector<byte>(get_length(fs));
    if (!read_values(fs, data)) return false;
    return true;
}

// Save a binary file
inline bool save_binary(const string& filename, const vector<byte>& data) {
    auto fs = open(filename.c_str(), "wb");
    if (!fs) return false;
    if (!write_values(fs, data)) return false;
    return true;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR CONCURRENCY UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms.
template <typename Func>
inline void parallel_for(
    int begin, int end, const Func& func, atomic<bool>* cancel, bool serial) {
    if (serial) {
        for (auto idx = begin; idx < end; idx++) {
            if (cancel && *cancel) break;
            func(idx);
        }
    } else {
        auto        threads  = vector<thread>{};
        auto        nthreads = thread::hardware_concurrency();
        atomic<int> next_idx(begin);
        for (auto thread_id = 0; thread_id < nthreads; thread_id++) {
            threads.emplace_back([&func, &next_idx, cancel, end]() {
                while (true) {
                    if (cancel && *cancel) break;
                    auto idx = next_idx.fetch_add(1);
                    if (idx >= end) break;
                    func(idx);
                }
            });
        }
        for (auto& t : threads) t.join();
    }
}

}  // namespace yocto

#endif
