#pragma once
// Android log.h mock for desktop builds
#include <stdarg.h>
#include <stdio.h>

#define ANDROID_LOG_VERBOSE 2
#define ANDROID_LOG_DEBUG   3
#define ANDROID_LOG_INFO    4
#define ANDROID_LOG_WARN    5
#define ANDROID_LOG_ERROR   6

static inline void __android_log_print(int prio, const char* tag, const char* fmt, ...) {
    (void)tag;
    FILE* out = (prio >= ANDROID_LOG_WARN) ? stderr : stdout;
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    fputc('\n', out);
}
