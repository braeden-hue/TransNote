/**
 * omr_engine.h  –  Public C API for the OMR inference engine.
 *
 * Designed for Dart FFI on Android and iOS.
 * All functions are thread-hostile: call from a single background isolate.
 *
 * Lifecycle:
 *   1. omr_create(paths)   – allocate engine, load models
 *   2. omr_process(...)    – run inference on one image
 *   3. omr_free_result(r)  – release memory returned by omr_process
 *   4. omr_destroy(engine) – release engine
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Opaque engine handle ─────────────────────────────────────────────────────
typedef void* OmrEngineHandle;

// ─── Token as plain C struct (Dart FFI safe) ──────────────────────────────────
typedef struct {
    int32_t  id;          // vocabulary index
    char     text[64];    // null-terminated token string, e.g. "note-C4-1/4"
} OmrTokenC;

// ─── Result returned to Dart ──────────────────────────────────────────────────
typedef struct {
    int         success;        // 1 = OK, 0 = error
    char        error[256];     // error message when success==0
    OmrTokenC*  tokens;         // heap-allocated token array
    int32_t     token_count;
} OmrResultC;

// ─── Lifecycle ────────────────────────────────────────────────────────────────

/**
 * Allocate and initialise the engine.
 * @param segnet_path    Path to segnet TFLite model
 * @param encoder_path   Path to encoder TFLite model
 * @param decoder_path   Path to decoder TFLite model
 * @param tokenizer_path Path to tokenizer.json
 * @return Opaque handle, or NULL on failure.
 */
OmrEngineHandle omr_create(const char* segnet_path,
                            const char* encoder_path,
                            const char* decoder_path,
                            const char* tokenizer_path);

/**
 * Run the full OMR pipeline on a raw image.
 * @param engine      Handle returned by omr_create
 * @param image_data  Raw image bytes (JPEG or PNG)
 * @param image_len   Byte length of image_data
 * @return Heap-allocated result; caller must pass to omr_free_result.
 */
OmrResultC* omr_process(OmrEngineHandle engine,
                         const uint8_t*  image_data,
                         int32_t         image_len);

/**
 * Free memory owned by an OmrResultC returned by omr_process.
 */
void omr_free_result(OmrResultC* result);

/**
 * Destroy the engine and release all resources.
 */
void omr_destroy(OmrEngineHandle engine);

#ifdef __cplusplus
} // extern "C"
#endif
