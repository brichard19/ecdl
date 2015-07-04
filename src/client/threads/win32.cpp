#include "threads.h"

Thread::Thread()
{
}

Thread::Thread(DWORD (*routine)(LPVOID), void *arg)
{
	this->handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)routine, arg, 0, &this->id);

	if( thread == NULL ) {
        throw "Error creating thread";	
    }
}

void Thread::kill()
{
	CloseHandle(this->handle);
}

void Thread::wait()
{
    WaitForSingleObject(this->handle, INFINITE);
}




Mutex::Mutex()
{
    this->handle = CreateMutex(NULL, FALSE, NULL);
}

Mutex::~Mutex()
{
    CloseHandle(this->handle);
}

void Mutex::grab()
{
	if(WaitForSingleObject( mutex->handle, INFINITE) != 0) {
        throw "Error waiting for mutex";	
    }
}

void Mutex::release()
{
    ReleaseMutex(this->handle); 
}

