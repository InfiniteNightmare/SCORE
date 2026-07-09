#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace score {
namespace update_io {

inline std::string atomic_temp_path(const std::string& path) {
    return path + ".tmp." + std::to_string((uint64_t)::getpid());
}

inline std::string parent_dir(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

inline void fsync_fd_or_throw(int fd, const std::string& path) {
    if (::fsync(fd) != 0) {
        int saved = errno;
        ::close(fd);
        throw std::runtime_error("fsync failed for " + path + ": " + std::strerror(saved));
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("close failed after fsync: " + path);
    }
}

inline void fsync_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("cannot open for fsync: " + path +
                                 " (" + std::strerror(errno) + ")");
    }
    fsync_fd_or_throw(fd, path);
}

inline void fsync_directory(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("cannot open directory for fsync: " + path +
                                 " (" + std::strerror(errno) + ")");
    }
    fsync_fd_or_throw(fd, path);
}

inline void fsync_file_if_requested(const std::string& path, bool do_fsync) {
    if (do_fsync) fsync_file(path);
}

inline void fsync_directory_if_requested(const std::string& path, bool do_fsync) {
    if (do_fsync) fsync_directory(path);
}

inline void fsync_parent_dir_if_requested(const std::string& path, bool do_fsync) {
    if (do_fsync) fsync_directory(parent_dir(path));
}

inline void publish_atomic_file(const std::string& tmp_path,
                                const std::string& path,
                                const char* artifact_name,
                                bool durable_publish) {
    if (durable_publish) fsync_file(tmp_path);
    if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
        int saved = errno;
        ::unlink(tmp_path.c_str());
        throw std::runtime_error(std::string("failed publishing ") + artifact_name + ": " + path +
                                 " (" + std::strerror(saved) + ")");
    }
    fsync_parent_dir_if_requested(path, durable_publish);
}

class ScopedFileLock {
public:
    ScopedFileLock(const std::string& path,
                   int open_flags,
                   int flock_mode,
                   const char* label,
                   bool optional_if_missing = false)
        : path_(path), label_(label) {
        fd_ = ::open(path_.c_str(), open_flags, 0644);
        if (fd_ < 0 && optional_if_missing && errno == ENOENT) {
            fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
            if (fd_ < 0 && (errno == EACCES || errno == EROFS)) return;
        }
        if (fd_ < 0) {
            if (optional_if_missing && errno == ENOENT) return;
            throw std::runtime_error(std::string("cannot open ") + label_ + ": " + path_ +
                                     " (" + std::strerror(errno) + ")");
        }
        while (::flock(fd_, flock_mode) != 0) {
            if (errno == EINTR) continue;
            int saved = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("cannot acquire ") + label_ + ": " + path_ +
                                     " (" + std::strerror(saved) + ")");
        }
    }

    ~ScopedFileLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    ScopedFileLock(const ScopedFileLock&) = delete;
    ScopedFileLock& operator=(const ScopedFileLock&) = delete;

private:
    std::string path_;
    const char* label_ = "file lock";
    int fd_ = -1;
};

class UpdateWriterLock {
public:
    explicit UpdateWriterLock(const std::string& index_path)
        : lock_(index_path + "/update_writer.lock",
                O_CREAT | O_RDWR | O_CLOEXEC,
                LOCK_EX,
                "update writer lock") {}

private:
    ScopedFileLock lock_;
};

class UpdatePublishWriteLock {
public:
    explicit UpdatePublishWriteLock(const std::string& index_path)
        : lock_(index_path + "/update_publish.lock",
                O_CREAT | O_RDWR | O_CLOEXEC,
                LOCK_EX,
                "update publish lock") {}

private:
    ScopedFileLock lock_;
};

class UpdatePublishReadLock {
public:
    explicit UpdatePublishReadLock(const std::string& index_path)
        : lock_(index_path + "/update_publish.lock",
                O_RDONLY | O_CLOEXEC,
                LOCK_SH,
                "update publish read lock",
                true) {}

private:
    ScopedFileLock lock_;
};

} // namespace update_io
} // namespace score
