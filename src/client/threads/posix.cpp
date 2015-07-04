#include "threads.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

Thread::Thread()
{
}

Thread::Thread(THREAD_FUNCTION (*routine)(VOID_PTR), void *arg)
{
	int r = pthread_create(&this->handle, NULL, routine, arg);

	if(r != 0) {
        throw "Error creating new thread"; 
	}
}

Thread::~Thread()
{
}

void Thread::kill()
{
	int r = pthread_cancel(this->handle);
    if(r != 0) {
        throw "Error killing thread";
    }
}

void Thread::wait()
{
    pthread_join(this->handle, NULL);
}


Mutex::Mutex()
{
    if(pthread_mutex_init(&this->handle, NULL)) {
        throw "Error creating mutex";
    }
}

void Mutex::destroy()
{
    pthread_mutex_destroy(&this->handle);
}

Mutex::~Mutex()
{
}

void Mutex::grab()
{
    if(pthread_mutex_lock(&this->handle)) {
        printf("Error locking mutex: %s\n", strerror(errno));
    }
}

void Mutex::release()
{
    if(pthread_mutex_unlock(&this->handle)) {
        printf("Error unlocking mutex: %s\n", strerror(errno));
    }
}
