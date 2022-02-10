#pragma once

#include <mutex>

typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> Lock;

class TryLock {
public:
    TryLock (Mutex& m, bool autoLock = true) : _mutex (m), _locked (false)
    {
        if (autoLock)
            _locked = _mutex.try_lock();
    }

    ~TryLock ()
    {
        if (_locked)
            _mutex.unlock();
    }

    bool acquire ()
    {
        _locked = _mutex.try_lock();
        return _locked;
    }

    void release ()
    {
        _mutex.unlock();
        _locked = false;
    }

    bool locked ()
    {
        return _locked;
    }

private:
    Mutex &		_mutex;
    bool		_locked;
};

class MutexPool {
public:
    MutexPool(size_t size = 256);
    ~MutexPool();

    Mutex &get(const void *address);

    static MutexPool *instance();

private:
    typedef Mutex *	MutexPtr;

    Mutex		mutex_;
    MutexPtr *	mutexes_;
    size_t		size_;
};
