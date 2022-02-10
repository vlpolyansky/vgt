#include "thread_mutex.h"
#include <cassert>

#define MUTEX_POOL_CHECK_FOR_DEADLOCKS 0

MutexPool global_mutexpool;

MutexPool::MutexPool(size_t size)
{
    size_		= size;
    mutexes_	= new MutexPtr[size];
    for (size_t k = 0; k < size; k++)
        mutexes_[k] = 0;
}

MutexPool::~MutexPool()
{
    for (size_t k = 0; k < size_; k++) {
        delete mutexes_[k];
        mutexes_[k] = 0;
    }
    delete[] mutexes_;
}

MutexPool *MutexPool::instance()
{
    return &global_mutexpool;
}

Mutex &MutexPool::get(const void *address)
{
    Lock lock(mutex_);

    size_t index = int(((size_t)(void *)(address) >> (sizeof(address) >> 1)) % size_);

#if MUTEX_POOL_CHECK_FOR_DEADLOCKS
    index = 0;
#endif

    Mutex *m = mutexes_[index];

    if (!m) {
        mutexes_[index] = new Mutex;
        m = mutexes_[index];
    }

    return *m;
}
