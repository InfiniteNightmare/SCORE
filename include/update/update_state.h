#pragma once

#include "update/update_io.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace score {

class LiveTable {
public:
    static std::shared_ptr<const LiveTable> empty(uint32_t max_pid) {
        auto table = std::make_shared<LiveTable>();
        table->max_pid_ = max_pid;
        table->live_count_ = 0;
        table->words_.assign(word_count(max_pid), 0);
        return table;
    }

    static std::shared_ptr<const LiveTable> all_live(uint32_t max_pid) {
        auto table = std::make_shared<LiveTable>();
        table->max_pid_ = max_pid;
        table->live_count_ = max_pid;
        table->words_.assign(word_count(max_pid), ~uint64_t{0});
        table->mask_tail();
        return table;
    }

    static std::shared_ptr<const LiveTable> from_live_pids(
        uint32_t max_pid,
        const std::vector<uint32_t>& live_pids) {
        auto table = std::make_shared<LiveTable>();
        table->max_pid_ = max_pid;
        table->live_count_ = 0;
        table->words_.assign(word_count(max_pid), 0);
        for (uint32_t pid : live_pids) {
            if (pid >= max_pid) throw std::out_of_range("live pid is outside LiveTable range");
            table->set_bit(pid, true);
        }
        return table;
    }

    std::shared_ptr<const LiveTable> with_updates(uint32_t new_max_pid,
                                                  const std::vector<uint32_t>& set_live,
                                                  const std::vector<uint32_t>& set_dead) const {
        if (new_max_pid < max_pid_) {
            throw std::invalid_argument("LiveTable updates cannot shrink max_pid");
        }
        auto table = std::make_shared<LiveTable>();
        table->max_pid_ = new_max_pid;
        table->live_count_ = live_count_;
        table->words_ = words_;
        table->words_.resize(word_count(new_max_pid), 0);
        table->mask_tail();

        for (uint32_t pid : set_dead) {
            if (pid >= new_max_pid) throw std::out_of_range("dead pid is outside LiveTable range");
            table->set_bit(pid, false);
        }
        for (uint32_t pid : set_live) {
            if (pid >= new_max_pid) throw std::out_of_range("live pid is outside LiveTable range");
            table->set_bit(pid, true);
        }
        return table;
    }

    void apply_updates_in_place(uint32_t new_max_pid,
                                const std::vector<uint32_t>& set_live,
                                const std::vector<uint32_t>& set_dead) {
        if (new_max_pid < max_pid_) {
            throw std::invalid_argument("LiveTable updates cannot shrink max_pid");
        }
        max_pid_ = new_max_pid;
        words_.resize(word_count(new_max_pid), 0);
        mask_tail();

        for (uint32_t pid : set_dead) {
            if (pid >= new_max_pid) throw std::out_of_range("dead pid is outside LiveTable range");
            set_bit(pid, false);
        }
        for (uint32_t pid : set_live) {
            if (pid >= new_max_pid) throw std::out_of_range("live pid is outside LiveTable range");
            set_bit(pid, true);
        }
    }

    bool is_live(uint32_t pid) const {
        if (pid >= max_pid_) return false;
        return (words_[(size_t)pid / 64] & bit_mask(pid)) != 0;
    }

    bool range_all_live(uint32_t begin_pid, uint32_t end_pid) const {
        if (begin_pid > end_pid || end_pid > max_pid_) return false;
        if (begin_pid == end_pid) return true;
        size_t first_word = (size_t)begin_pid / 64;
        size_t last_word = (size_t)(end_pid - 1) / 64;
        uint64_t first_mask = ~uint64_t{0} << (begin_pid & 63u);
        uint32_t end_offset = end_pid & 63u;
        uint64_t last_mask = end_offset == 0
            ? ~uint64_t{0}
            : ((uint64_t{1} << end_offset) - 1);
        if (first_word == last_word) {
            uint64_t mask = first_mask & last_mask;
            return (words_[first_word] & mask) == mask;
        }
        if ((words_[first_word] & first_mask) != first_mask) return false;
        for (size_t word = first_word + 1; word < last_word; ++word) {
            if (words_[word] != ~uint64_t{0}) return false;
        }
        return (words_[last_word] & last_mask) == last_mask;
    }

    uint32_t live_count() const { return live_count_; }
    uint32_t max_pid() const { return max_pid_; }
    const std::vector<uint64_t>& words() const { return words_; }

    void save_binary(const std::string& path, bool durable_publish = false) const {
        const std::string tmp_path = update_io::atomic_temp_path(path);
        try {
            {
                std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                if (!out.is_open()) throw std::runtime_error("cannot create LiveTable checkpoint: " + tmp_path);
                uint32_t magic = 0x3142564cu; // "LVB1" in little-endian native order.
                uint32_t version = 1;
                uint64_t word_count_u64 = (uint64_t)words_.size();
                out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                out.write(reinterpret_cast<const char*>(&version), sizeof(version));
                out.write(reinterpret_cast<const char*>(&max_pid_), sizeof(max_pid_));
                out.write(reinterpret_cast<const char*>(&live_count_), sizeof(live_count_));
                out.write(reinterpret_cast<const char*>(&word_count_u64), sizeof(word_count_u64));
                if (!words_.empty()) {
                    out.write(reinterpret_cast<const char*>(words_.data()), words_.size() * sizeof(uint64_t));
                }
                out.flush();
                if (!out) throw std::runtime_error("failed writing LiveTable checkpoint: " + tmp_path);
            }
            update_io::publish_atomic_file(tmp_path, path, "LiveTable checkpoint", durable_publish);
        } catch (...) {
            ::unlink(tmp_path.c_str());
            throw;
        }
    }

    static std::shared_ptr<const LiveTable> load_binary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("cannot open LiveTable checkpoint: " + path);
        auto read_pod = [&](auto& value, const char* name) {
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!in) throw std::runtime_error(std::string("short LiveTable checkpoint while reading ") + name + ": " + path);
        };
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t max_pid = 0;
        uint32_t live_count = 0;
        uint64_t word_count_u64 = 0;
        read_pod(magic, "magic");
        read_pod(version, "version");
        read_pod(max_pid, "max_pid");
        read_pod(live_count, "live_count");
        read_pod(word_count_u64, "word_count");
        if (magic != 0x3142564cu || version != 1) {
            throw std::runtime_error("invalid LiveTable checkpoint header: " + path);
        }
        if (word_count_u64 != (uint64_t)word_count(max_pid) ||
            word_count_u64 > (uint64_t)std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("LiveTable checkpoint word count mismatch: " + path);
        }
        auto table = std::make_shared<LiveTable>();
        table->max_pid_ = max_pid;
        table->live_count_ = live_count;
        table->words_.resize((size_t)word_count_u64);
        if (!table->words_.empty()) {
            in.read(reinterpret_cast<char*>(table->words_.data()), table->words_.size() * sizeof(uint64_t));
            if (!in) throw std::runtime_error("short LiveTable checkpoint payload: " + path);
        }
        table->mask_tail();
        uint32_t counted = 0;
        for (uint64_t word : table->words_) counted += (uint32_t)__builtin_popcountll(word);
        if (counted != live_count) {
            throw std::runtime_error("LiveTable checkpoint live_count mismatch: " + path);
        }
        return table;
    }

private:
    static size_t word_count(uint32_t max_pid) {
        return ((size_t)max_pid + 63) / 64;
    }

    static uint64_t bit_mask(uint32_t pid) {
        return uint64_t{1} << (pid & 63u);
    }

    void set_bit(uint32_t pid, bool live) {
        uint64_t& word = words_[(size_t)pid / 64];
        uint64_t mask = bit_mask(pid);
        bool was_live = (word & mask) != 0;
        if (live == was_live) return;
        if (live) {
            word |= mask;
            ++live_count_;
        } else {
            word &= ~mask;
            --live_count_;
        }
    }

    void mask_tail() {
        size_t tail_bits = (size_t)max_pid_ & 63u;
        if (tail_bits != 0 && !words_.empty()) {
            words_.back() &= (uint64_t{1} << tail_bits) - 1;
        }
    }

    uint32_t max_pid_ = 0;
    uint32_t live_count_ = 0;
    std::vector<uint64_t> words_;
};

class SparseDirectMapTable {
public:
    size_t size() const { return size_; }

    void reserve(size_t expected_size) {
        size_t min_capacity = capacity_for_size(expected_size);
        if (min_capacity <= keys_.size()) return;
        rehash(min_capacity);
    }

    bool lookup(uint32_t key, uint32_t& value_out) const {
        if (keys_.empty()) return false;
        size_t slot = find_existing_slot(key);
        if (slot == npos()) return false;
        value_out = values_[slot];
        return true;
    }

    bool insert_new(uint32_t key, uint32_t value) {
        ensure_capacity(size_ + 1);
        auto [slot, found] = find_slot_for_insert(key);
        if (found) return false;
        occupy_slot(slot, key, value);
        return true;
    }

    bool set(uint32_t key, uint32_t value) {
        return set_with_old(key, value, nullptr);
    }

    bool set_with_old(uint32_t key, uint32_t value, uint32_t* old_value_out) {
        if (keys_.empty()) {
            rehash(capacity_for_size(1));
        } else if ((size_ + 1 + deleted_) * 4 > keys_.size() * 3) {
            size_t existing_slot = find_existing_slot(key);
            if (existing_slot != npos()) {
                if (old_value_out) *old_value_out = values_[existing_slot];
                values_[existing_slot] = value;
                return false;
            }
            rehash(capacity_for_size(size_ + 1));
        }
        auto [slot, found] = find_slot_for_insert(key);
        if (found) {
            if (old_value_out) *old_value_out = values_[slot];
            values_[slot] = value;
            return false;
        }
        occupy_slot(slot, key, value);
        return true;
    }

    bool erase(uint32_t key, uint32_t* value_out = nullptr) {
        if (keys_.empty()) return false;
        size_t slot = find_existing_slot(key);
        if (slot == npos()) return false;
        if (value_out) *value_out = values_[slot];
        states_[slot] = kDeleted;
        --size_;
        ++deleted_;
        if (deleted_ > size_ && keys_.size() > kMinCapacity) {
            rehash(capacity_for_size(size_));
        }
        return true;
    }

    template<typename Fn>
    void for_each(Fn&& fn) const {
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (states_[i] == kOccupied) fn(keys_[i], values_[i]);
        }
    }

private:
    static constexpr uint8_t kEmpty = 0;
    static constexpr uint8_t kOccupied = 1;
    static constexpr uint8_t kDeleted = 2;
    static constexpr size_t kMinCapacity = 16;

    static size_t npos() { return std::numeric_limits<size_t>::max(); }

    static uint64_t mix_key(uint32_t key) {
        uint64_t z = (uint64_t)key + 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    static size_t next_power_of_two(size_t value) {
        size_t out = kMinCapacity;
        while (out < value) {
            if (out > std::numeric_limits<size_t>::max() / 2) {
                throw std::overflow_error("SparseDirectMap capacity overflow");
            }
            out *= 2;
        }
        return out;
    }

    static size_t capacity_for_size(size_t expected_size) {
        if (expected_size == 0) return 0;
        if (expected_size > (std::numeric_limits<size_t>::max() / 4) * 3) {
            throw std::overflow_error("SparseDirectMap size overflow");
        }
        return next_power_of_two((expected_size * 4 + 2) / 3);
    }

    void ensure_capacity(size_t wanted_size) {
        if (keys_.empty()) {
            rehash(capacity_for_size(wanted_size));
            return;
        }
        if ((wanted_size + deleted_) * 4 > keys_.size() * 3) {
            rehash(capacity_for_size(std::max(wanted_size, size_ + 1)));
        }
    }

    size_t find_existing_slot(uint32_t key) const {
        size_t mask = keys_.size() - 1;
        size_t slot = (size_t)mix_key(key) & mask;
        while (true) {
            uint8_t state = states_[slot];
            if (state == kEmpty) return npos();
            if (state == kOccupied && keys_[slot] == key) return slot;
            slot = (slot + 1) & mask;
        }
    }

    std::pair<size_t, bool> find_slot_for_insert(uint32_t key) const {
        size_t mask = keys_.size() - 1;
        size_t slot = (size_t)mix_key(key) & mask;
        size_t first_deleted = npos();
        while (true) {
            uint8_t state = states_[slot];
            if (state == kEmpty) {
                return {first_deleted == npos() ? slot : first_deleted, false};
            }
            if (state == kDeleted) {
                if (first_deleted == npos()) first_deleted = slot;
            } else if (keys_[slot] == key) {
                return {slot, true};
            }
            slot = (slot + 1) & mask;
        }
    }

    void occupy_slot(size_t slot, uint32_t key, uint32_t value) {
        if (states_[slot] == kDeleted) --deleted_;
        states_[slot] = kOccupied;
        keys_[slot] = key;
        values_[slot] = value;
        ++size_;
    }

    void rehash(size_t new_capacity) {
        if (new_capacity == 0) {
            keys_.clear();
            values_.clear();
            states_.clear();
            size_ = 0;
            deleted_ = 0;
            return;
        }
        new_capacity = next_power_of_two(new_capacity);
        std::vector<uint32_t> old_keys = std::move(keys_);
        std::vector<uint32_t> old_values = std::move(values_);
        std::vector<uint8_t> old_states = std::move(states_);
        keys_.assign(new_capacity, 0);
        values_.assign(new_capacity, 0);
        states_.assign(new_capacity, kEmpty);
        size_ = 0;
        deleted_ = 0;
        for (size_t i = 0; i < old_keys.size(); ++i) {
            if (old_states[i] != kOccupied) continue;
            bool inserted = insert_new(old_keys[i], old_values[i]);
            (void)inserted;
        }
    }

    std::vector<uint32_t> keys_;
    std::vector<uint32_t> values_;
    std::vector<uint8_t> states_;
    size_t size_ = 0;
    size_t deleted_ = 0;
};

class DirectMap {
public:
    static constexpr uint32_t kMissingPid = std::numeric_limits<uint32_t>::max();

    enum class Mode {
        Dense,
        Sparse
    };

    static DirectMap dense(uint32_t initial_max_orig_id = 0) {
        DirectMap map;
        map.mode_ = Mode::Dense;
        map.dense_.assign((size_t)initial_max_orig_id + 1, kMissingPid);
        return map;
    }

    static DirectMap sparse() {
        DirectMap map;
        map.mode_ = Mode::Sparse;
        return map;
    }

    Mode mode() const { return mode_; }
    size_t live_size() const { return live_size_; }

    void save_binary(const std::string& path, bool durable_publish = false) const {
        const std::string tmp_path = update_io::atomic_temp_path(path);
        try {
            {
                std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                if (!out.is_open()) throw std::runtime_error("cannot create DirectMap: " + tmp_path);
                uint32_t magic = 0x314d4453u; // "SDM1" in little-endian native order.
                uint32_t version = 1;
                uint32_t mode = mode_ == Mode::Dense ? 1u : 2u;
                uint64_t live = (uint64_t)live_size_;
                out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                out.write(reinterpret_cast<const char*>(&version), sizeof(version));
                out.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
                out.write(reinterpret_cast<const char*>(&live), sizeof(live));
                if (mode_ == Mode::Dense) {
                    uint64_t count = (uint64_t)dense_.size();
                    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                    if (!dense_.empty()) {
                        out.write(reinterpret_cast<const char*>(dense_.data()),
                                  dense_.size() * sizeof(uint32_t));
                    }
                } else {
                    uint64_t count = (uint64_t)sparse_.size();
                    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                    sparse_.for_each([&](uint32_t orig, uint32_t pid) {
                        out.write(reinterpret_cast<const char*>(&orig), sizeof(uint32_t));
                        out.write(reinterpret_cast<const char*>(&pid), sizeof(uint32_t));
                    });
                }
                out.flush();
                if (!out) throw std::runtime_error("failed writing DirectMap: " + tmp_path);
            }
            update_io::publish_atomic_file(tmp_path, path, "DirectMap", durable_publish);
        } catch (...) {
            ::unlink(tmp_path.c_str());
            throw;
        }
    }

    static DirectMap load_binary(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("cannot open DirectMap: " + path);
        auto read_pod = [&](auto& value, const char* name) {
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!in) throw std::runtime_error(std::string("short DirectMap while reading ") + name + ": " + path);
        };
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t mode = 0;
        uint64_t stored_live = 0;
        read_pod(magic, "magic");
        read_pod(version, "version");
        read_pod(mode, "mode");
        read_pod(stored_live, "live_size");
        if (magic != 0x314d4453u || version != 1 || (mode != 1 && mode != 2)) {
            throw std::runtime_error("invalid DirectMap header: " + path);
        }
        uint64_t count = 0;
        read_pod(count, "count");
        if (count > (uint64_t)std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("DirectMap count exceeds size_t range: " + path);
        }

        DirectMap map;
        map.mode_ = mode == 1 ? Mode::Dense : Mode::Sparse;
        if (map.mode_ == Mode::Dense) {
            map.dense_.resize((size_t)count, kMissingPid);
            if (count != 0) {
                in.read(reinterpret_cast<char*>(map.dense_.data()),
                        (size_t)count * sizeof(uint32_t));
                if (!in) throw std::runtime_error("short DirectMap dense payload: " + path);
            }
            map.live_size_ = 0;
            for (uint32_t pid : map.dense_) {
                if (pid != kMissingPid) ++map.live_size_;
            }
        } else {
            if (count > (uint64_t)std::numeric_limits<size_t>::max() / sizeof(std::pair<uint32_t, uint32_t>)) {
                throw std::runtime_error("DirectMap sparse count is too large: " + path);
            }
            map.sparse_.reserve((size_t)count);
            for (uint64_t i = 0; i < count; ++i) {
                uint32_t orig = 0;
                uint32_t pid = 0;
                read_pod(orig, "sparse orig_id");
                read_pod(pid, "sparse pid");
                if (pid == kMissingPid) {
                    throw std::runtime_error("DirectMap sparse payload uses missing sentinel as PID: " + path);
                }
                if (!map.sparse_.insert_new(orig, pid)) {
                    throw std::runtime_error("DirectMap sparse payload contains duplicate orig_id: " + path);
                }
            }
            map.live_size_ = map.sparse_.size();
        }
        if (map.live_size_ != (size_t)stored_live) {
            throw std::runtime_error("DirectMap live_size mismatch: " + path);
        }
        return map;
    }

    void reserve_entries(size_t count) {
        if (mode_ == Mode::Sparse) sparse_.reserve(count);
    }

    bool lookup(uint32_t orig_id, uint32_t& pid_out) const {
        if (mode_ == Mode::Dense) {
            if ((size_t)orig_id >= dense_.size()) return false;
            uint32_t pid = dense_[(size_t)orig_id];
            if (pid == kMissingPid) return false;
            pid_out = pid;
            return true;
        }
        return sparse_.lookup(orig_id, pid_out);
    }

    bool contains(uint32_t orig_id) const {
        uint32_t ignored = 0;
        return lookup(orig_id, ignored);
    }

    void set(uint32_t orig_id, uint32_t pid) {
        (void)set_with_old_pid(orig_id, pid, nullptr);
    }

    bool set_with_old_pid(uint32_t orig_id, uint32_t pid, uint32_t* old_pid_out) {
        if (pid == kMissingPid) {
            throw std::invalid_argument("UINT32_MAX is reserved as DirectMap missing sentinel");
        }
        if (mode_ == Mode::Dense) {
            if ((size_t)orig_id >= dense_.size()) {
                dense_.resize((size_t)orig_id + 1, kMissingPid);
            }
            uint32_t old_pid = dense_[(size_t)orig_id];
            bool had_old = old_pid != kMissingPid;
            if (had_old) {
                if (old_pid_out) *old_pid_out = old_pid;
            } else {
                ++live_size_;
            }
            dense_[(size_t)orig_id] = pid;
            return had_old;
        }
        uint32_t old_pid = 0;
        bool inserted = sparse_.set_with_old(orig_id, pid, &old_pid);
        if (inserted) {
            ++live_size_;
            return false;
        }
        if (old_pid_out) *old_pid_out = old_pid;
        return true;
    }

    bool erase(uint32_t orig_id, uint32_t* pid_out = nullptr) {
        if (mode_ == Mode::Dense) {
            if ((size_t)orig_id >= dense_.size()) return false;
            uint32_t pid = dense_[(size_t)orig_id];
            if (pid == kMissingPid) return false;
            if (pid_out) *pid_out = pid;
            dense_[(size_t)orig_id] = kMissingPid;
            --live_size_;
            return true;
        }
        if (sparse_.erase(orig_id, pid_out)) {
            --live_size_;
            return true;
        }
        return false;
    }

    std::vector<uint32_t> live_pids_sorted_by_orig() const {
        std::vector<std::pair<uint32_t, uint32_t>> entries = live_entries_sorted_by_orig();
        std::vector<uint32_t> pids;
        pids.reserve(entries.size());
        for (const auto& kv : entries) pids.push_back(kv.second);
        return pids;
    }

    template<typename Fn>
    void for_each_live_entry(Fn&& fn) const {
        if (mode_ == Mode::Dense) {
            for (size_t orig = 0; orig < dense_.size(); ++orig) {
                uint32_t pid = dense_[orig];
                if (pid != kMissingPid) fn((uint32_t)orig, pid);
            }
            return;
        }
        sparse_.for_each(std::forward<Fn>(fn));
    }

    std::vector<std::pair<uint32_t, uint32_t>> live_entries_sorted_by_orig() const {
        std::vector<std::pair<uint32_t, uint32_t>> entries;
        entries.reserve(live_size_);
        if (mode_ == Mode::Dense) {
            for (size_t orig = 0; orig < dense_.size(); ++orig) {
                uint32_t pid = dense_[orig];
                if (pid != kMissingPid) entries.push_back({(uint32_t)orig, pid});
            }
        } else {
            sparse_.for_each([&](uint32_t orig, uint32_t pid) {
                entries.push_back({orig, pid});
            });
        }
        std::sort(entries.begin(), entries.end());
        return entries;
    }

private:
    Mode mode_ = Mode::Dense;
    std::vector<uint32_t> dense_;
    SparseDirectMapTable sparse_;
    size_t live_size_ = 0;
};

} // namespace score
