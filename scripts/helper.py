
def parseInt(n):
        return int(n, 0)

def read_int(prompt):
    s = raw_input(prompt)
    s.rstrip("0x")
    
    return parseInt(s);

def read_string(prompt):
    return raw_input(prompt)

def main():
    
    p = read_int("p:")

    a = read_int("a:")

    b = read_int("b:")

    n = read_int("n:")
    
    gx = read_int("Gx:")

    gy = read_int("Gy:")

    qx = read_int("Qx:")

    qy = read_int("Qy:")

    dBits = read_int("distinguished bits:")

    filename = read_string("Filename:")

    data = {}
    data['params'] = {}
    data['params']['field'] = "prime"
    data['params']['a'] = str(a)
    data['params']['b'] = str(b)
    data['params']['p'] = str(p)
    data['params']['n'] = str(n)
    data['params']['gx'] = str(gx)
    data['params']['gy'] = str(gy)
    data['params']['qx'] = str(qx)
    data['params']['qy'] = str(qy)
    data['params']['bits'] = dBits

    dataStr = str(data).replace("\'", "\"")
    f = file(filename, 'w')
    f.write(dataStr)
    f.close()

if __name__ == "__main__":
    main()
