#ifndef VIDI_FILEMAP_H
#define VIDI_FILEMAP_H

#if !defined(_WIN32)
#include <sys/mman.h>
#else
#include <windows.h>
#endif

#include <deque>
#include <algorithm>
#include <string>
#include <memory>
#include <atomic>
#include <stdexcept>

#include <cassert>
#include <cstring>

#ifndef INVALID_HANDLE_VALUE
#define INVALID_HANDLE_VALUE -1
#endif

namespace vidi {

#if defined(_WIN32)  // Platform: Windows
typedef HANDLE FileDesc;
#else  // Platform: POSIX
typedef int FileDesc;
#endif // defined(_WIN32)

struct FileRef;
typedef std::shared_ptr<FileRef> FileMap;

enum struct FileFlag { 
  READ, WRITE, READ_DIRECT, WRITE_DIRECT, 
};

// ----------------------------------------------------------------------------
// File I/O
// ----------------------------------------------------------------------------
void ThrowLastError(const std::string &msg);

struct FutureBuffer
{
 protected:
  const char *ptr{nullptr};
  size_t size;

 public:
  virtual ~FutureBuffer() {}
  FutureBuffer(size_t length, const void *target) : ptr((const char *)target), size(length) {}
  const char * get() const { return ptr; }
  size_t numOfBytes() const { return size; }
  virtual bool ready() const { return true; }
  virtual void wait() const { ; }
  virtual void cancel(FileDesc) const { ; }
  virtual void cancel(FileMap) const { ; }
};

typedef std::shared_ptr<FutureBuffer> future_buffer_t;

template <typename T>
future_buffer_t makeBasicFutureBuffer(T &&x)
{
  return std::make_shared<FutureBuffer>(std::forward<T>(x));
}

template <typename... Args>
future_buffer_t makeBasicFutureBuffer(Args &&... args)
{
  return std::make_shared<FutureBuffer>(std::forward<Args>(args)...);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------

struct FileRef
{
  FileDesc fd = INVALID_HANDLE_VALUE;

 protected:
  const static bool verbose = false;
  const size_t file_size = 0;
  const FileFlag flag;

 public:
  virtual ~FileRef();
  FileRef(const std::string &filename, FileDesc h, FileFlag flag);
  FileRef(const std::string &filename, FileDesc h, size_t size, FileFlag flag);
  size_t getFileSize() const { return file_size; }

  virtual future_buffer_t randomRead(size_t offset, size_t bytes, void *data) = 0;
  virtual future_buffer_t randomWrite(size_t offset, size_t bytes, const void *data) = 0;
};

// struct FileRef_Sequential : virtual FileRef
// {
//   virtual void setFilePointer(size_t offset) = 0;
//   virtual size_t getFilePointer() = 0;
//   virtual future_buffer_t readData(void *data, size_t bytes, size_t _unused) = 0;
//   virtual future_buffer_t writeData(const void *data, size_t bytes, size_t _unused) = 0;
// };

// struct FileRef_Random : virtual FileRef
// {
//   virtual future_buffer_t randomRead(size_t offset, size_t bytes, void *data) = 0;
//   virtual future_buffer_t randomWrite(size_t offset, size_t bytes, const void *data) = 0;
// };

// ----------------------------------------------------------------------------
//
// File I/O using virtual memories (mmap)
//
// ----------------------------------------------------------------------------

struct FileRef_VM : public FileRef
{
#if defined(_WIN32)
  HANDLE hMap = INVALID_HANDLE_VALUE;
  char *map = NULL;
#else  // Platform: Unix
  char *map = (char *)MAP_FAILED;
#endif  // defined(_WIN32)

 public:
  ~FileRef_VM();
  /* reader */ FileRef_VM(const std::string &filename);
  /* writer */ FileRef_VM(const std::string &filename, size_t requested_size);
  future_buffer_t randomRead(size_t offset, size_t bytes, void *data) override;
  future_buffer_t randomWrite(size_t offset, size_t bytes, const void *data) override;
};

// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------

struct FileRef_Native : public FileRef
{
 public:
  /* reader */ FileRef_Native(const std::string &filename, FileFlag flag);
  /* writer */ FileRef_Native(const std::string &filename, size_t requested_size, FileFlag mode);
  future_buffer_t randomRead(size_t offset, size_t bytes, void *data) override;
  future_buffer_t randomWrite(size_t offset, size_t bytes, const void *data) override;
};

// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------

#if defined(_WIN32) 
struct FileRef_Async : public FileRef
{
  struct AsyncFutureBuffer : public FutureBuffer
  {
    mutable HANDLE hEvent = NULL;
    HANDLE hFile;
    LPOVERLAPPED overlapped_structure;

#ifndef NDEBUG // debug information
    uint64_t offset;
    uint64_t nbytes;
    uint64_t fsize;
#endif

   private:
    void create(size_t offset)
    {
      overlapped_structure = new OVERLAPPED();
      overlapped_structure->Internal = 0;
	    overlapped_structure->InternalHigh = 0;
	    overlapped_structure->Offset = offset & 0xFFFFFFFF;
	    overlapped_structure->OffsetHigh = offset >> 32;
      overlapped_structure->hEvent = hEvent;
    }

   public:
    AsyncFutureBuffer(HANDLE hFile, size_t offset, size_t bytes, const void *target)
        : FutureBuffer(bytes, target), hFile(hFile)
    {
      hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
      if (hEvent == NULL) {
        ThrowLastError("Could not CreateEvent");
      }

      create(offset);
    }

    ~AsyncFutureBuffer()
    {
      assert(ready());
      delete overlapped_structure;
      CloseHandle(hEvent);
    }

    bool ready() const override
    {
      return HasOverlappedIoCompleted(overlapped_structure);
    }

    void wait() const override
    {
      DWORD numOfBytesRead = 0;
      DWORD err = GetOverlappedResult(hFile, overlapped_structure, &numOfBytesRead, TRUE);
      if (!err) {
        ThrowLastError("async IO failed");
      }
      ResetEvent(hEvent);
      assert(numOfBytesRead == size);
    }

    void cancel(FileDesc fd) const override
    {
      BOOL result = CancelIoEx(fd, overlapped_structure);
      if (result == TRUE || GetLastError() != ERROR_NOT_FOUND) {
        return;
      }
      ThrowLastError("failed to cancel IO");
      // Wait for the I/O subsystem to acknowledge our cancellation.
      // Depending on the timing of the calls, the I/O might complete with a
      // cancellation status, or it might complete normally (if the ReadFile was
      // in the process of completing at the time CancelIoEx was called, or if
      // the device does not support cancellation).
      // This call specifies TRUE for the bWait parameter, which will block
      // until the I/O either completes or is canceled, thus resuming execution,
      // provided the underlying device driver and associated hardware are functioning
      // properly. If there is a problem with the driver it is better to stop
      // responding here than to try to continue while masking the problem.
    }

    void cancel(FileMap hfile) const override
    {
      cancel(hfile->fd);
    }
  };

 public:
  FileRef_Async(const std::string &filename);

  FileRef_Async(const std::string &filename, size_t requested_size);

  future_buffer_t readData(void *data, size_t bytes, size_t offset);

  future_buffer_t writeData(const void *data, size_t bytes, size_t offset);

  future_buffer_t randomRead(size_t offset, size_t bytes, void *data) override
  {
    return readData(data, bytes, offset);
  }

  future_buffer_t randomWrite(size_t offset, size_t bytes, const void *data) override
  {
    return writeData(data, bytes, offset);
  }
};
#endif

// ----------------------------------------------------------------------------
//
// API
//
// ----------------------------------------------------------------------------

inline FileMap filemap_write_create(const std::string &filename, size_t requested_size)
{
  return std::make_shared<FileRef_VM>(filename, requested_size);
  // return std::make_shared<FileRef_Async>(filename, requested_size);
  // return std::make_shared<FileRef_Native>(filename, requested_size, FileFlag::WRITE);
}

inline FileMap filemap_read_create(const std::string &filename)
{
  return std::make_shared<FileRef_VM>(filename);
  // return std::make_shared<FileRef_Async>(filename);
  // return std::make_shared<FileRef_Native>(filename, FileFlag::READ);
}

inline FileMap filemap_write_create_direct(const std::string &filename)
{
  throw std::runtime_error("not implemented");
}

inline FileMap filemap_read_create_direct(const std::string &filename)
{
  return std::make_shared<FileRef_Native>(filename, FileFlag::READ_DIRECT);
}

inline FileMap filemap_write_create_async(const std::string &filename, size_t requested_size)
{
#if defined(_WIN32)  // Platform: Windows
  // return std::make_shared<FileRef_VM>(filename, requested_size);
  return std::make_shared<FileRef_Async>(filename, requested_size);
#else
  return std::make_shared<FileRef_VM>(filename, requested_size);
#endif
}

inline FileMap filemap_read_create_async(const std::string &filename)
{
#if defined(_WIN32)  // Platform: Windows
  // return std::make_shared<FileRef_VM>(filename);
  return std::make_shared<FileRef_Async>(filename);
#else
  return std::make_shared<FileRef_VM>(filename);
#endif
}

inline void filemap_close(FileMap &file)
{
  file.reset();
}

/* deprecated */
// inline void filemap_write(FileMap &file, const void *data, const size_t bytes)
// {
//   auto r = file->writeData(data, bytes, size_t(-1));
//   r->wait();
// }

// inline void filemap_read(FileMap &file, void *data, const size_t bytes)
// {
//   auto r = file->readData(data, bytes, size_t(-1));
//   r->wait();
// }

/* synchronous */
inline void filemap_random_read(FileMap &file, size_t offset, void *data, const size_t bytes)
{
  const size_t batch = 0x80000000U;
  std::deque<future_buffer_t> queue;
  for (size_t o = 0; o < bytes; o += batch) {
    queue.push_back(file->randomRead(offset + o, std::min(batch, bytes - o), ((uint8_t *)data) + o));
  }
  while (!queue.empty()) {
    queue.front()->wait();
    queue.pop_front();
  }
}

inline void filemap_random_write(FileMap &file, size_t offset, const void *data, const size_t bytes)
{
  const size_t batch = 0x80000000U;
  std::deque<future_buffer_t> queue;
  for (size_t o = 0; o < bytes; o += batch) {
    queue.push_back(file->randomWrite(offset + o, std::min(batch, bytes - o), ((uint8_t *)data) + o));
  }
  while (!queue.empty()) {
    queue.front()->wait();
    queue.pop_front();
  }
}

inline void filemap_random_write_update(FileMap &file,
                                        size_t &offset,
                                        const void *data,
                                        const size_t bytes)
{
  filemap_random_write(file, offset, data, bytes);
  offset += bytes;
}

inline void filemap_random_read_update(FileMap &file,
                                       size_t &offset,
                                       void *data,
                                       const size_t bytes)
{
  filemap_random_read(file, offset, data, bytes);
  offset += bytes;
}

/* asynchronous */
inline future_buffer_t filemap_random_read_async(FileMap &file,
                                                 size_t offset,
                                                 void *data,
                                                 const size_t bytes)
{
  return file->randomRead(offset, bytes, data);
}

inline future_buffer_t filemap_random_write_async(FileMap &file,
                                                  size_t offset,
                                                  void *data,
                                                  const size_t bytes)
{
  return file->randomWrite(offset, bytes, data);
}

}  // namespace vidi

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// #ifdef VIDI_FILEMAP_IMPLEMENTATION

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// #include <stdio.h>  // For printf()
// #include <string.h>  // For memcmp()
// #include <stdlib.h>  // For exit()

namespace vidi {

inline void ThrowLastError(const std::string &msg)
{
#if defined(_WIN32)  // Create a string with last error message
  auto result = std::string("no error");
  DWORD error = GetLastError();
  if (error) {
    LPVOID lpMsgBuf;
    DWORD bufLen = FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0,
        NULL);
    if (bufLen) {
      LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
      result = lpMsgStr, lpMsgStr + bufLen;
      LocalFree(lpMsgBuf);
    }
  }
  printf("%s: %s\n", msg.c_str(), result.c_str());
  throw std::runtime_error(msg + ": " + result);
#else
  perror(msg.c_str());
  throw std::runtime_error("Termination caused by I/O errors");
#endif  // defined(_WIN32)
}

// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------
static size_t read_file_size(const std::string &filename, FileDesc h)
{
  if (h == INVALID_HANDLE_VALUE) {
    ThrowLastError("failed to map file " + filename);
  }
#if defined(_WIN32)  // Platform: Windows
  LARGE_INTEGER size;
  GetFileSizeEx(h, &size);
  return size.QuadPart;
#else  // Platform: POSIX
  struct stat size = {0};
  if (fstat(h, &size) == -1) {
    ThrowLastError("error getting the file size");
  }
  return size.st_size;
#endif  // defined(_WIN32)
}

static void stretch_file_size(FileDesc h, size_t requested_size)
{
#if defined(_WIN32)  // Platform: Windows
  // Stretch the file size to the size of the (mapped) array of char
  LARGE_INTEGER file_size;
  file_size.QuadPart = requested_size;
  if (!SetFilePointerEx(h, file_size, NULL, FILE_BEGIN)) {
    ThrowLastError("error calling SetFilePointerEx() to 'stretch' the file");
  }

  // Actually stretch the file
  SetEndOfFile(h);

  // Verify file size
  GetFileSizeEx(h, &file_size);
  if (file_size.QuadPart != requested_size) {
    ThrowLastError("incorrect file size");
  }

#else  // Platform: POSIX

  // Stretch the file size to the size of the (mmapped) array of char
  if (::lseek(h, requested_size - 1, SEEK_SET) == -1) {
    ThrowLastError("error calling lseek() to 'stretch' the file");
  }

  /* Something needs to be written at the end of the file to
     have the file actually have the new size.
     Just writing an empty string at the current file position will do.
     Note:
     - The current position in the file is at the end of the stretched
       file due to the call to lseek().
     - An empty string is actually a single '\0' character, so a zero-byte
       will be written at the last byte of the file. */

  if (::write(h, "", 1) == -1) {
    ThrowLastError("error writing last byte of the file");
  }

#endif  // Platform: Windows
}

inline FileRef::FileRef(const std::string &filename, FileDesc h, FileFlag mode)
    : fd(h), file_size(read_file_size(filename, h)), flag(mode)
{
  assert(mode == FileFlag::READ || mode == FileFlag::READ_DIRECT);
  if (verbose)
    printf("file size is %zu\n", file_size);
}

inline FileRef::FileRef(const std::string &filename, FileDesc h, size_t size, FileFlag mode)
    : fd(h), file_size(size), flag(mode)
{
  assert(mode == FileFlag::WRITE || mode == FileFlag::WRITE_DIRECT);
  if (verbose)
    printf("file size is %zu\n", file_size);
}

inline FileRef::~FileRef()
{
#if defined(_WIN32)  // Platform: Windows
  FlushFileBuffers(this->fd);
  CloseHandle(this->fd);
#else  // Platform: POSIX
  // Un-mmaping doesn't close the file, so we still need to do that.
  close(this->fd);
#endif  // defined(_WIN32)
}

// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------

inline FileRef_VM::~FileRef_VM()
{
#if defined(_WIN32)  // Platform: Windows
  UnmapViewOfFile(this->map);
  CloseHandle(this->hMap);
#else  // Platform: POSIX
  // Don't forget to free the mmapped memory
  if (munmap(this->map, this->file_size) == -1) {
    ThrowLastError("error un-mmapping the file");
  }
#endif  // defined(_WIN32)
}

/* reader */
inline FileRef_VM::FileRef_VM(const std::string &filename)
    : FileRef(filename,
#if defined(_WIN32)  // Platform: Windows
              CreateFile(filename.c_str(),
                         FILE_ATTRIBUTE_READONLY,
                         FILE_SHARE_READ,
                         NULL,
                         OPEN_EXISTING,
                         FILE_FLAG_RANDOM_ACCESS,
                         NULL),
#else
              open(filename.c_str(), O_RDONLY, (mode_t)0600),
#endif
              FileFlag::READ)
{
  if (this->file_size == 0) {
    throw std::runtime_error("cannot map 0 size file");
  }

  // Now map the file to virtual memory
#if defined(_WIN32)  // Platform: Windows
  this->hMap = CreateFileMapping(this->fd, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (this->hMap == INVALID_HANDLE_VALUE) {
    ThrowLastError("failed to map file " + filename);
  }
  this->map = (char *)MapViewOfFile(this->hMap, FILE_MAP_READ, 0, 0, this->file_size);
  if (!this->map) {
    ThrowLastError("failed to map file " + filename);
  }
#else  // Platform: POSIX
  this->map = (char *)mmap(0, this->file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (this->map == MAP_FAILED) {
    ThrowLastError("failed to map of file " + filename);
  }
#endif  // defined(_WIN32)
}

/* writer */
inline FileRef_VM::FileRef_VM(const std::string &filename, size_t requested_size)
    : FileRef(filename,
#if defined(_WIN32)  // Platform: Windows
              CreateFile(filename.c_str(),
                         GENERIC_WRITE | GENERIC_READ,
                         FILE_SHARE_WRITE,
                         NULL,
                         CREATE_ALWAYS,
                         FILE_FLAG_WRITE_THROUGH,
                         NULL),
#else
              /* Open a file for writing. Note: "O_WRONLY" mode is not sufficient when mmaping.
               - Creating the file if it doesn't exist.
               - Truncating it to 0 size if it already exists. (not really needed) */
              open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600),
#endif
              requested_size,
              FileFlag::WRITE)
{
  if (this->fd == INVALID_HANDLE_VALUE) {
    ThrowLastError("failed to open file " + filename);
  }
  stretch_file_size(this->fd, requested_size);

#if defined(_WIN32)  // Platform: Windows
  // Now map the file
  this->hMap = CreateFileMapping(this->fd, nullptr, PAGE_READWRITE, 0, 0, nullptr);
  if (this->hMap == INVALID_HANDLE_VALUE) {
    ThrowLastError("failed to map file " + filename);
  }
  this->map = (char *)MapViewOfFile(this->hMap, FILE_MAP_ALL_ACCESS, 0, 0, requested_size);
  if (!this->map) {
    ThrowLastError("failed to map of file " + filename);
  }
#else  // Platform: POSIX
  // Now the file is ready to be mmapped.
  this->map = (char *)mmap(0, requested_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  if (this->map == MAP_FAILED) {
    ThrowLastError("failed to map of file " + filename);
  }
#endif  // defined(_WIN32)
}

inline future_buffer_t FileRef_VM::randomRead(size_t offset, size_t bytes, void *data)
{
  assert(this->flag == FileFlag::READ);
  assert(bytes <= this->file_size);

  if (verbose) { printf("read %zu bytes\n", bytes); }

  std::memcpy((char *)data, this->map + offset, bytes);

  return makeBasicFutureBuffer(bytes, data);
}

inline future_buffer_t FileRef_VM::randomWrite(size_t offset, size_t bytes, const void *data)
{
  assert(this->flag == FileFlag::WRITE);
  assert(bytes <= this->file_size);

  if (verbose) { printf("writing %zu bytes\n", bytes); }

  std::memcpy(this->map + offset, (char *)data, bytes);

  return makeBasicFutureBuffer(bytes, data);
}

// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------

inline FileRef_Native::FileRef_Native(const std::string& filename, FileFlag flag)
  : FileRef(filename,
#if defined(_WIN32)
            CreateFile(filename.c_str(),
                       FILE_ATTRIBUTE_READONLY,
                       FILE_SHARE_READ,
                       NULL,
                       OPEN_EXISTING,
                       /* The file or device attributes and flags */
                       flag == FileFlag::READ ? (FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS) : FILE_FLAG_NO_BUFFERING,
                       NULL),
#else
            open(filename.c_str(), O_RDONLY, (mode_t)0600),
#endif
            flag)
{
  if (this->fd == INVALID_HANDLE_VALUE)
    ThrowLastError("failed to open file " + filename);
  if (this->file_size == 0)
    ThrowLastError("error: file is empty, nothing to do");
}

inline FileRef_Native::FileRef_Native(const std::string& filename, size_t requested_size, FileFlag flag)
  : FileRef(filename,
#if defined(_WIN32) // Platform: Windows
            CreateFile(filename.c_str(), GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_FLAG_WRITE_THROUGH, NULL),
#else
            /* Open a file for writing. Note: "O_WRONLY" mode is not sufficient when mmaping.
             - Creating the file if it doesn't exist.
             - Truncating it to 0 size if it already exists. (not really needed) */
            open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600),
#endif
            requested_size,
            FileFlag::WRITE)
{
  if (this->fd == INVALID_HANDLE_VALUE) {
    ThrowLastError("failed to open file " + filename);
  }
  stretch_file_size(this->fd, requested_size);
}

inline future_buffer_t FileRef_Native::randomRead(size_t offset, size_t bytes, void *data)
{
#if defined(_WIN32)
  LARGE_INTEGER large;
  large.QuadPart = offset;
  OVERLAPPED ol{0};
  ol.Offset = large.LowPart;
  ol.OffsetHigh = large.HighPart;
  DWORD br;
  if (!ReadFile(this->fd, data, bytes, &br, &ol))
    ThrowLastError("failed to load streaming buffer");
#else
  const ssize_t br = pread(this->fd, data, bytes, offset);
  if (br == -1)
    ThrowLastError("failed to load streaming buffer");
#endif
  assert(br == bytes);
  return makeBasicFutureBuffer(bytes, data);
}

inline future_buffer_t FileRef_Native::randomWrite(size_t offset, size_t bytes, const void *data)
{
  throw std::runtime_error("write is not supported");
}

// ----------------------------------------------------------------------------
//
//
//
// ----------------------------------------------------------------------------
#if defined(_WIN32)
inline FileRef_Async::FileRef_Async(const std::string &filename)
    : FileRef(filename,
              CreateFile(filename.c_str(),
                         GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ,
                         NULL,
                         OPEN_EXISTING,
                         FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                         NULL),
              FileFlag::READ_DIRECT)
{
  if (this->fd == INVALID_HANDLE_VALUE) {
    ThrowLastError("failed to open file " + filename);
  }

  if (this->file_size == 0) {
    ThrowLastError("error: file is empty, nothing to do");
  }
}

inline FileRef_Async::FileRef_Async(const std::string &filename, size_t requested_size)
    : FileRef(filename,
              CreateFile(filename.c_str(),
                         GENERIC_WRITE | GENERIC_READ,
                         FILE_SHARE_WRITE,
                         NULL,
                         CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                         NULL),
              requested_size,
              FileFlag::WRITE)
{
}

inline future_buffer_t FileRef_Async::readData(void *data, size_t bytes, size_t offset)
{
  assert(bytes <= this->file_size);
  assert(offset != size_t(-1));
  auto ret = std::make_shared<AsyncFutureBuffer>(this->fd, offset, bytes, data);

#ifndef NDEBUG 
  ret->fsize = this->file_size;
  ret->offset = offset;
  ret->nbytes = bytes;
#endif

  DWORD numOfBytesRead = 0;
  if (!ReadFile(this->fd, data, bytes, &numOfBytesRead, ret->overlapped_structure)) {
    if (GetLastError() != ERROR_IO_PENDING) {
      // Some other error occurred while reading the file.
      ThrowLastError("failed to start async read");
    }
    else {
      // Operation has been queued and
      // will complete in the future.
    }
    return ret;
  }
  else {
    // Operation has completed immediately.
    assert(numOfBytesRead == bytes);
    return makeBasicFutureBuffer(bytes, data);
  }
}

inline future_buffer_t FileRef_Async::writeData(const void *data, size_t bytes, size_t offset)
{
  assert(bytes <= this->file_size);
  assert(offset != size_t(-1));
  auto ret = std::make_shared<AsyncFutureBuffer>(this->fd, offset, bytes, data);

  DWORD numOfBytesRead = 0;
  if (!WriteFile(this->fd, data, bytes, &numOfBytesRead, ret->overlapped_structure)) {
    if (GetLastError() != ERROR_IO_PENDING) {
      // Some other error occurred while reading the file.
      ThrowLastError("failed to start async write");
    }
    else {
      // Operation has been queued and
      // will complete in the future.
    }
    return ret;
  }
  else {
    // Operation has completed immediately.
    assert(numOfBytesRead == bytes);
    return makeBasicFutureBuffer(bytes, data);
  }
}
#endif

}  // namespace vidi

// #endif  // VIDI_FILEMAP_IMPLEMENTATION

#endif  // VIDI_FILEMAP_H
