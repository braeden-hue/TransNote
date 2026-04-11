// Android AAssetManager desktop mock implementation
#include "android/asset_manager.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct AAssetManager {
    std::string base_dir;
};

struct AAsset {
    std::vector<unsigned char> data;
    size_t pos{0};
};

extern "C" {

AAssetManager* desktop_create_asset_manager(const char* base_dir) {
    auto* mgr = new AAssetManager();
    mgr->base_dir = base_dir;
    return mgr;
}

void desktop_destroy_asset_manager(AAssetManager* mgr) {
    delete mgr;
}

AAsset* AAssetManager_open(AAssetManager* mgr, const char* filename, int /*mode*/) {
    std::string path = mgr->base_dir + "/" + filename;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[asset_manager] cannot open: %s\n", path.c_str());
        return nullptr;
    }
    auto* asset = new AAsset();
    asset->data.assign(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
    return asset;
}

off_t AAsset_getLength(AAsset* asset) {
    return static_cast<off_t>(asset->data.size());
}

int AAsset_read(AAsset* asset, void* buf, size_t count) {
    size_t remaining = asset->data.size() - asset->pos;
    size_t n = (count < remaining) ? count : remaining;
    std::memcpy(buf, asset->data.data() + asset->pos, n);
    asset->pos += n;
    return static_cast<int>(n);
}

void AAsset_close(AAsset* asset) {
    delete asset;
}

} // extern "C"
