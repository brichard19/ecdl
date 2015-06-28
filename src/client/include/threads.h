#ifndef _THREADS_H
#define _THREADS_H

#ifdef WIN32
/* Use windows threading */

#include<windows.h>

typedef DWORD THREAD_FUNCTION;
typedef LPVOID VOID_PTR;

#else

/* Use POSIX threading */
#include<pthread.h>

typedef void* THREAD_FUNCTION;
typedef void* VOID_PTR;
#endif


class Thread {
private:
#ifdef WIN32
	HANDLE handle;
	DWORD id;		
#else
    pthread_t handle;
#endif

public:

    Thread();
    Thread( THREAD_FUNCTION (*routine)(VOID_PTR), void *arg );
    ~Thread();
    void wait();
    void kill();
};

class Mutex {
private:
#ifdef WIN32
	HANDLE handle;
#else
    pthread_mutex_t handle;
#endif

public:
    Mutex();
    ~Mutex();
    void grab();
    void release();
    void destroy();
};

#endif
