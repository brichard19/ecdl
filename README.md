# ecdl

This is an implementation of the parallel Pollard's rho algorithm, applied to the elliptic curve discrete logarithm problem.

It solves the ECDLP for curves over a prime field, in Weierstrass form (Y^2 = X^3 + aX + b)

It consists of a central server program and a client program. The client program can be run on many machines to help solve
the problem faster. The client requests work from the server and reports any distinguished points it finds.
The server collects the points and stores them in a database. When two colliding points are found, the problem
can be solved.

It is still a very new project and probalby has many bugs.

#### Dependencies

Currently builds on Linux using make.

Client:

* g++
* GNU Multiprecision Arithmetic Library ([https://gmplib.org/](https://gmplib.org/))
* libcurl ([http://curl.haxx.se/](http://curl.haxx.se/))

Server:
* python 2.7 with flask and MySQLdb
* mysql database

Optional:
* nasm for building with the optimized x86 routines
* sage ([http://www.sagemath.org](http://www.sagemath.org)) for running script to generate ECDL problems

### Building the client

To build the CPU client, run `make client_cpu` in the `src/client` directory

```
# make client_cpu
```

There is a GPU client using CUDA, but it is currently broken because it only accepts values in montgomery form. The rest of the code was recently switched to use the Barrett reduction, so they are incompatible.


#### Running the server

There is a server config file `config/config.json` which needs to be edited

```
{
    "port":9999,                 // Port the server listens on
    "dbUser":"user",             // mysql user name
    "dbPassword":"password",     // mysql user password
    "dbHost":"127.0.0.1"         // mysql host
}
```

The server is run using the `python` command

```
# python server.py
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

    "point_cache_size": 128,                 // Points to collect before sending to server
    "cpu_threads": 4,                       // Number of threads. 1 thread per core is optimal
    "cpu_points_per_thread": 1024           // Number of points each thread will compute in parallel
}
```


After a job has been set up on the server, the client can be run. It takes the job name as its argument:

```
# ./client-cpu ecp56
```


#### Solving

Currently there is a manual step involved. This will be fixed in the future.

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

To check if this is a solution, plug these values into the `solve.py` program.

```
# python python solve.py 0x55bd5461e01ade 0x3a1c06b8843ff7 0x2ab615a84ce098 0x39fe7a74c5175a 0x64355cd6200000 0x71d04a7b43448 ecp56
```

The output will look something like this:
```
Counting walk 1 length
Found endpoint
391036
Counting walk 2 length
Found endpoint
297364
Checking for Robin Hood
Not a Robin Hood :)
Stepping 93672 times
Searching for collision
Found collision!
0x3547e3e36752e8 0x25d6f02438168 0x1a8454e115340bL 0x9df3a3469de720L
0x90ce1fd597aa72 0xd25379c576bfe 0x1a8454e115340bL 0x9df3a3469de720L
Q=   0x5136e72ea9c95a 0x999eb1108ab4f
kG = 0x5136e72ea9c95aL 0x999eb1108ab4fL

k=0x77fc86a17007L
```
#### Choosing the number of distinguished bits

The number of distinguished bits determines the trade-off between space and running time of the algorithm.

The value to choose depends on the amount of storage available, number of processors, and speed of the processors.

A naive collision search on a curve of order 2^n requires 2^(n/2) storage and (2^(n/2))/m time for m processors.

Using the distinguished point technique, the search requires (2^(n/2))/(2^(n/2-d)) storage and ((2^(n/2))/m + 2.5 * 2^d)t time for d
distinguished bits and m processors, and t is the time for a single point addition. Note that this is including the time it takes for
the collison to be detected and solved using the script above.

For example, to solve the discrete logarithm on a curve with an order of ~2^80, it would require about 2^40 points to find a collision.

If using a 24-bit distinguisher, then you will need to find about 2^16 distinguished points. On 128 processors where each processor can do 

1 million point additions per second, the running time would be approximately (2^32 / + 2.5 * 2^24)0.000001 = 1.2 hours.

