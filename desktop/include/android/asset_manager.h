#pragma once
// Android asset_manager.h mock for desktop builds
#include <stddef.h>
#include <sys/types.h>

#define AASSET_MODE_UNKNOWN   0
#define AASSET_MODE_RANDOM    1
#define AASSET_MODE_STREAMING 2
#define AASSET_MODE_BUFFER    3

struct AAssetManager;
struct AAsset;

#ifdef __cplusplus
extern "C" {
#endif

AAssetManager* desktop_create_asset_manager(const char* base_dir);
void           desktop_destroy_asset_manager(AAssetManager* mgr);

AAsset* AAssetManager_open(AAssetManager* mgr, const char* filename, int mode);
off_t   AAsset_getLength(AAsset* asset);
int     AAsset_read(AAsset* asset, void* buf, size_t count);
void    AAsset_close(AAsset* asset);

#ifdef __cplusplus
}
#endif
