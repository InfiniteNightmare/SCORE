#pragma once

#include "update/update_io.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace score {

struct UpdateManifestDelta {
    std::string segment_id;
    std::string path;
    uint32_t pid_begin = 0;
    uint32_t pid_count = 0;
    uint32_t live_count = 0;
    uint64_t artifact_bytes = 0;
};

struct UpdateManifest {
    std::string format;
    uint64_t generation = 0;
    std::string base_path;
    uint32_t base_n = 0;
    uint32_t max_pid = 0;
    uint32_t live_count = 0;
    std::vector<UpdateManifestDelta> sealed_deltas;
    std::string wal_path;
    std::string checkpoint_path;
    std::string direct_map_mode;
    std::string direct_map_path;
    bool direct_map_optional_for_search = true;
    std::string durability_mode;
    uint64_t last_committed_lsn = 0;
    uint64_t last_fsynced_lsn = 0;
};

inline bool is_valid_update_durability_mode(const std::string& mode) {
    return mode == "strict" || mode == "buffered" || mode == "none";
}

inline bool update_durability_fsync_each_commit(const std::string& mode) {
    return mode == "strict";
}

inline bool update_durability_flush_at_buffer_boundary(const std::string& mode) {
    return mode == "buffered";
}

inline bool update_durability_fsync_artifacts(const std::string& mode) {
    return mode != "none";
}

inline std::string update_manifest_path(const std::string& index_path) {
    return index_path + "/update_manifest.json";
}

inline bool update_manifest_exists(const std::string& index_path) {
    const std::string path = update_manifest_path(index_path);
    if (::access(path.c_str(), F_OK) == 0) return true;
    if (errno == ENOENT) return false;
    throw std::runtime_error("cannot probe update manifest: " + path);
}

namespace impl {

inline const rapidjson::Value& require_json_member(const rapidjson::Value& object,
                                                   const char* name,
                                                   const std::string& path) {
    if (!object.IsObject()) {
        throw std::runtime_error("update manifest expected object while reading " + std::string(name) +
                                 ": " + path);
    }
    auto it = object.FindMember(name);
    if (it == object.MemberEnd()) {
        throw std::runtime_error("update manifest missing field '" + std::string(name) + "': " + path);
    }
    return it->value;
}

inline std::string require_json_string(const rapidjson::Value& object,
                                       const char* name,
                                       const std::string& path) {
    const rapidjson::Value& value = require_json_member(object, name, path);
    if (!value.IsString()) {
        throw std::runtime_error("update manifest field '" + std::string(name) +
                                 "' must be a string: " + path);
    }
    return value.GetString();
}

inline uint64_t require_json_uint64(const rapidjson::Value& object,
                                    const char* name,
                                    const std::string& path) {
    const rapidjson::Value& value = require_json_member(object, name, path);
    if (!value.IsUint64()) {
        throw std::runtime_error("update manifest field '" + std::string(name) +
                                 "' must be an unsigned integer: " + path);
    }
    return value.GetUint64();
}

inline uint32_t require_json_uint32(const rapidjson::Value& object,
                                    const char* name,
                                    const std::string& path) {
    uint64_t value = require_json_uint64(object, name, path);
    if (value > (uint64_t)std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("update manifest field '" + std::string(name) +
                                 "' exceeds uint32 range: " + path);
    }
    return (uint32_t)value;
}

inline bool optional_json_bool(const rapidjson::Value& object,
                               const char* name,
                               bool default_value,
                               const std::string& path) {
    if (!object.IsObject()) {
        throw std::runtime_error("update manifest expected object while reading " + std::string(name) +
                                 ": " + path);
    }
    auto it = object.FindMember(name);
    if (it == object.MemberEnd()) return default_value;
    if (!it->value.IsBool()) {
        throw std::runtime_error("update manifest field '" + std::string(name) +
                                 "' must be a boolean: " + path);
    }
    return it->value.GetBool();
}

inline uint64_t optional_json_uint64(const rapidjson::Value& object,
                                     const char* name,
                                     uint64_t default_value,
                                     const std::string& path) {
    if (!object.IsObject()) {
        throw std::runtime_error("update manifest expected object while reading " + std::string(name) +
                                 ": " + path);
    }
    auto it = object.FindMember(name);
    if (it == object.MemberEnd()) return default_value;
    if (!it->value.IsUint64()) {
        throw std::runtime_error("update manifest field '" + std::string(name) +
                                 "' must be an unsigned integer: " + path);
    }
    return it->value.GetUint64();
}

inline void append_json_string(std::string& out, const std::string& value) {
    static constexpr char kHex[] = "0123456789abcdef";
    out.push_back('"');
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    out += "\\u00";
                    out.push_back(kHex[ch >> 4]);
                    out.push_back(kHex[ch & 0x0f]);
                } else {
                    out.push_back((char)ch);
                }
                break;
        }
    }
    out.push_back('"');
}

inline void append_json_key(std::string& out, const char* key) {
    out.push_back('"');
    out += key;
    out += "\":";
}

inline std::string render_update_manifest_json(const UpdateManifest& manifest) {
    std::string out;
    out.reserve(512 + manifest.sealed_deltas.size() * 160);
    out.push_back('{');
    append_json_key(out, "format");
    append_json_string(out, manifest.format);
    out += ",\"generation\":";
    out += std::to_string(manifest.generation);

    out += ",\"base\":{\"path\":";
    append_json_string(out, manifest.base_path);
    out += ",\"base_n\":";
    out += std::to_string(manifest.base_n);
    out.push_back('}');

    out += ",\"pid\":{\"base_n\":";
    out += std::to_string(manifest.base_n);
    out += ",\"max_pid\":";
    out += std::to_string(manifest.max_pid);
    out += ",\"live_count\":";
    out += std::to_string(manifest.live_count);
    out.push_back('}');

    out += ",\"sealed_deltas\":[";
    for (size_t i = 0; i < manifest.sealed_deltas.size(); ++i) {
        if (i != 0) out.push_back(',');
        const UpdateManifestDelta& delta = manifest.sealed_deltas[i];
        out += "{\"segment_id\":";
        append_json_string(out, delta.segment_id);
        out += ",\"path\":";
        append_json_string(out, delta.path);
        out += ",\"pid_begin\":";
        out += std::to_string(delta.pid_begin);
        out += ",\"pid_count\":";
        out += std::to_string(delta.pid_count);
        out += ",\"live_count\":";
        out += std::to_string(delta.live_count);
        out += ",\"artifact_bytes\":";
        out += std::to_string(delta.artifact_bytes);
        out.push_back('}');
    }
    out.push_back(']');

    out += ",\"growing\":{\"wal_path\":";
    append_json_string(out, manifest.wal_path);
    out += ",\"checkpoint_path\":";
    append_json_string(out, manifest.checkpoint_path);
    out.push_back('}');

    out += ",\"direct_map\":{\"mode\":";
    append_json_string(out, manifest.direct_map_mode);
    out += ",\"path\":";
    append_json_string(out, manifest.direct_map_path);
    out += ",\"optional_for_search\":";
    out += manifest.direct_map_optional_for_search ? "true" : "false";
    out.push_back('}');

    out += ",\"durability\":{\"mode\":";
    append_json_string(out, manifest.durability_mode);
    out += ",\"last_committed_lsn\":";
    out += std::to_string(manifest.last_committed_lsn);
    out += ",\"last_fsynced_lsn\":";
    out += std::to_string(manifest.last_fsynced_lsn);
    out += "}}";
    return out;
}

} // namespace impl

inline UpdateManifest load_update_manifest_json(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open update manifest: " + path);
    }

    rapidjson::IStreamWrapper stream(in);
    rapidjson::Document doc;
    doc.ParseStream(stream);
    if (doc.HasParseError()) {
        throw std::runtime_error(
            "invalid update manifest JSON at byte " +
            std::to_string((uint64_t)doc.GetErrorOffset()) + ": " +
            rapidjson::GetParseError_En(doc.GetParseError()) + ": " + path);
    }
    if (!doc.IsObject()) {
        throw std::runtime_error("update manifest root must be an object: " + path);
    }

    UpdateManifest manifest;
    manifest.format = impl::require_json_string(doc, "format", path);
    if (manifest.format != "score_update_manifest_v1") {
        throw std::runtime_error("unsupported update manifest format '" + manifest.format + "': " + path);
    }
    manifest.generation = impl::require_json_uint64(doc, "generation", path);

    const rapidjson::Value& base = impl::require_json_member(doc, "base", path);
    manifest.base_path = impl::require_json_string(base, "path", path);
    manifest.base_n = impl::require_json_uint32(base, "base_n", path);

    const rapidjson::Value& pid = impl::require_json_member(doc, "pid", path);
    uint32_t pid_base_n = impl::require_json_uint32(pid, "base_n", path);
    if (pid_base_n != manifest.base_n) {
        throw std::runtime_error("update manifest base.base_n and pid.base_n differ: " + path);
    }
    manifest.max_pid = impl::require_json_uint32(pid, "max_pid", path);
    manifest.live_count = impl::require_json_uint32(pid, "live_count", path);
    if (manifest.max_pid < manifest.base_n) {
        throw std::runtime_error("update manifest max_pid is smaller than base_n: " + path);
    }
    if (manifest.live_count > manifest.max_pid) {
        throw std::runtime_error("update manifest live_count exceeds max_pid: " + path);
    }

    const rapidjson::Value& sealed = impl::require_json_member(doc, "sealed_deltas", path);
    if (!sealed.IsArray()) {
        throw std::runtime_error("update manifest sealed_deltas must be an array: " + path);
    }
    manifest.sealed_deltas.reserve(sealed.Size());
    uint64_t sealed_live_sum = 0;
    for (rapidjson::SizeType i = 0; i < sealed.Size(); ++i) {
        const rapidjson::Value& item = sealed[i];
        if (!item.IsObject()) {
            throw std::runtime_error("update manifest sealed_deltas entry must be an object: " + path);
        }
        UpdateManifestDelta delta;
        delta.segment_id = impl::require_json_string(item, "segment_id", path);
        delta.path = impl::require_json_string(item, "path", path);
        delta.pid_begin = impl::require_json_uint32(item, "pid_begin", path);
        delta.pid_count = impl::require_json_uint32(item, "pid_count", path);
        delta.live_count = impl::require_json_uint32(item, "live_count", path);
        delta.artifact_bytes = impl::optional_json_uint64(item, "artifact_bytes", 0, path);
        if (delta.segment_id.empty()) {
            throw std::runtime_error("update manifest delta segment_id must not be empty: " + path);
        }
        if (delta.path.empty()) {
            throw std::runtime_error("update manifest delta path must not be empty: " + path);
        }
        if (delta.pid_count == 0) {
            throw std::runtime_error("update manifest delta pid_count must be positive: " + path);
        }
        if (delta.live_count > delta.pid_count) {
            throw std::runtime_error("update manifest delta live_count exceeds pid_count: " + path);
        }
        if (delta.pid_begin < manifest.base_n) {
            throw std::runtime_error("update manifest delta PID range overlaps base segment: " + path);
        }
        if ((uint64_t)delta.pid_begin + (uint64_t)delta.pid_count > (uint64_t)manifest.max_pid) {
            throw std::runtime_error("update manifest delta PID range exceeds max_pid: " + path);
        }
        sealed_live_sum += delta.live_count;
        if (sealed_live_sum > manifest.live_count) {
            throw std::runtime_error("update manifest sealed delta live_count exceeds global live_count: " + path);
        }
        for (const UpdateManifestDelta& existing : manifest.sealed_deltas) {
            if (existing.segment_id == delta.segment_id) {
                throw std::runtime_error("update manifest contains duplicate delta segment_id: " + path);
            }
            if (existing.path == delta.path) {
                throw std::runtime_error("update manifest contains duplicate delta path: " + path);
            }
        }
        manifest.sealed_deltas.push_back(std::move(delta));
    }

    const rapidjson::Value& growing = impl::require_json_member(doc, "growing", path);
    manifest.wal_path = impl::require_json_string(growing, "wal_path", path);
    manifest.checkpoint_path = impl::require_json_string(growing, "checkpoint_path", path);

    const rapidjson::Value& direct_map = impl::require_json_member(doc, "direct_map", path);
    manifest.direct_map_mode = impl::require_json_string(direct_map, "mode", path);
    manifest.direct_map_path = impl::require_json_string(direct_map, "path", path);
    manifest.direct_map_optional_for_search =
        impl::optional_json_bool(direct_map, "optional_for_search", true, path);
    if (manifest.direct_map_mode != "dense" && manifest.direct_map_mode != "sparse") {
        throw std::runtime_error("update manifest direct_map.mode must be dense or sparse: " + path);
    }

    const rapidjson::Value& durability = impl::require_json_member(doc, "durability", path);
    manifest.durability_mode = impl::require_json_string(durability, "mode", path);
    if (!is_valid_update_durability_mode(manifest.durability_mode)) {
        throw std::runtime_error("update manifest durability.mode is invalid: " + path);
    }
    manifest.last_committed_lsn = impl::require_json_uint64(durability, "last_committed_lsn", path);
    manifest.last_fsynced_lsn = impl::require_json_uint64(durability, "last_fsynced_lsn", path);
    if (manifest.last_fsynced_lsn > manifest.last_committed_lsn) {
        throw std::runtime_error("update manifest fsynced LSN is ahead of committed LSN: " + path);
    }
    if (update_durability_fsync_each_commit(manifest.durability_mode) &&
        manifest.last_fsynced_lsn < manifest.last_committed_lsn) {
        throw std::runtime_error("update manifest fsynced LSN is behind committed LSN: " + path);
    }

    return manifest;
}

inline UpdateManifest load_update_manifest_json_for_index(const std::string& index_path) {
    return load_update_manifest_json(update_manifest_path(index_path));
}

inline void write_update_manifest_json(const UpdateManifest& manifest,
                                       const std::string& path,
                                       bool durable_publish = false) {
    std::string tmp_path = update_io::atomic_temp_path(path);
    try {
        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot create update manifest: " + tmp_path);
            std::string json = impl::render_update_manifest_json(manifest);
            out.write(json.data(), (std::streamsize)json.size());
            out.flush();
            if (!out) throw std::runtime_error("failed writing update manifest: " + tmp_path);
        }
        update_io::publish_atomic_file(tmp_path, path, "update manifest", durable_publish);
    } catch (...) {
        ::unlink(tmp_path.c_str());
        throw;
    }
}

inline void write_update_manifest_json_for_index(const UpdateManifest& manifest,
                                                 const std::string& index_path,
                                                 bool durable_publish = false) {
    write_update_manifest_json(manifest, update_manifest_path(index_path), durable_publish);
}

} // namespace score
