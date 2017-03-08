# ecdl

This is an implementation of the parallel Pollard's rho algorithm, applied to the elliptic curve discrete logarithm problem.

It solves the ECDLP for curves over a prime field, in Weierstrass form `Y^2 = X^3 + aX + b`

It consists of a central server program and a client program. The client program can be run on many machines to help solve
the problem faster. The client requests work from the server and sends back the results.

It is not a mature project yet, so watch out for bugs.

#### Dependencies

Currently builds on Linux using make.

Client:

* g++
* GNU Multiprecision Arithmetic Library ([https://gmplib.org/](https://gmplib.org/))
* libcurl ([http://curl.haxx.se/](http://curl.haxx.se/))

CUDA client:

* CUDA Toolkit ([https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit))

Server:
* mysql database
* python 2.7 with flask and MySQLdb

Optional:
* NASM for building with the optimized x86 routines
* sage ([http://www.sagemath.org](http://www.sagemath.org)) for running script to generate ECDL problems

### Building the client

To build the client, run the Makefile in the `src/client` directory.

To build the CPU client:

```
# make client_cpu
```

To build the CUDA client:

```
# make client_cuda
```


#### Running the server

The server consists of two programs: `server.py` and `solver.py`.

`server.py` is whats clients connect to when they request work.

`solver.py` waits until a collision is found in the database and then attempts to solve the discrete logarithm

There is a server config file `config/config.json` which needs to be edited

```
{
    "port":9999,                 // Port the server listens on
    "dbUser":"user",             // mysql user name
    "dbPassword":"password",     // mysql user password
    "dbHost":"127.0.0.1"         // mysql host
}
```

```
Both programs are run using `python`:
```


```
# python server.py
```

```
# python solver.py
```



#### Setting up a job on the server

The parameters for the particular problem you want to solve are encded in JSON format

Values can be in decimal or hex with the `0x` prefix

```
params:{
    "field":"prime",                // Field used. Currently only "prime" is supported
    "a":"0x39C95E6DDDB1BC45733C",   // 'a' term in the curve equation
    "b":"0x1F16D880E89D5A1C0ED1 ",  // 'b' term in the curve equation
    "p":"0x62CE5177412ACA899CF5",   // prime modulus of the field
    "n":"0x62CE5177407B7258DC31",   // order of the curve
    "gx":"0x315D4B201C208475057D",  // 'x' value of generator point G
    "gy":"0x035F3DF5AB370252450A",  // 'y' value of generator point G
    "qx":"0x0679834CEFB7215DC365",  // 'x' value of point Q
    "qy":"0x4084BC50388C4E6FDFAB",  // 'y' value of point Q
    "bits":"20"                     // Number of distinguished bits (default is 20)
}
```

There is a sage script in the scripts directory can generate random parameters and write them to the file for you.

For example, to generate a curve with a 56-bit prime modulus, run:

```
# ./gencurve.sh 56 ecp56.json
```

Once you have your JSON file, you can upload it to the server using the create.sh script. As arguments it takes the json file and the name of the job

```
./create.sh ecp56.json ecp56
```

#### Running the client

There is a `settings.json` file that needs to be edited

```
{
    "server_host": "127.0.0.1",             // Server host
    "server_port": 9999,                    // server port

    "point_cache_size": 128,                // Points to collect before sending to server
    "cpu_threads": 4,                       // Number of threads. 1 thread per core is optimal
    "cpu_points_per_thread": 16             // Number of points each thread will compute in parallel


    "cuda_blocks": 1,                       // Number of CUDA blocks
    "cuda_threads": 32,                     // Number of CUDA threads per block
    "cuda_points_per_thread": 16,           // Number of points each CUDA thread will compute in parallel
    "cuda_device": 0                        // The index of the CUDA device to use
}
```

After a job has been set up on the server, the client can be run. It takes the job name as its argument:

```
# ./client-cpu ecp56
```


#### Solving

When the server finds two colliding distinguished points, it will output them to stdout like this:

```
==== FOUND COLLISION ====
a1: 0x55bd5461e01ade
b1: 0x3a1c06b8843ff7

a2: 0x2ab615a84ce098
b2: 0x39fe7a74c5175a

x: 0x64355cd6200000
y: 0x71d04a7b43448

```

If `solver.py` is running, it will detect the collision and attempt to find the solution.


The output will look something like this:
```
Counting walk #1 length
Found endpoint
1410541
Counting walk #2 length
Found endpoint
1380960
Checking for Robin Hood
Not a Robin Hood :)
Stepping 29581 times
Searching for collision
Found collision!
0x3547e3e36752e8 0x25d6f02438168 0x1a8454e115340b 0x9df3a3469de720
0x90ce1fd597aa72 0xd25379c576bfe 0x1a8454e115340b 0x9df3a3469de720
Verification successful
The solution is 77fc86a17007

```

#### Choosing the number of distinguished bits

The number of distinguished bits determines the trade-off between space and running time of the algorithm.

The value to choose depends on the amount of storage available, number of processors, and speed of the processors.

A naive collision search on a curve of order `2^n` requires `2^(n/2)` storage and `(2^(n/2))/m` time for `m` processors.

Using the distinguished point technique, the search requires `(2^(n/2-d)` storage and `((2^(n/2))/m + 2.5 * 2^d)t` time where `d` is the number of distinguished bits, `m` is the number of processors and `t` is the time it takes to perform 1 point addition. Note that this is including the time it takes for the collison to be detected and solved on the server.

For example, to solve the discrete logarithm on a curve with an order of `~2^80`, it would require about `2^40` points to find a collision.

If using a 24-bit distinguisher, then you will need to find about `2^16` distinguished points. On 128 processors where each processor can do 1 million point additions per second, the running time would be approximately `(2^40 / 128 + 2.5 * 2^24)0.000001` = 2.3 hours.

