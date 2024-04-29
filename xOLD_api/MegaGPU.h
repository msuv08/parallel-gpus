#ifndef MEGAGPU_H
#define MEGAGPU_H

class MegaGPU {
public:
    MegaGPU(int numElements);
    ~MegaGPU();
    void addArrays(const int *a, const int *b, int *c);

private:
    int *d_a0, *d_b0, *d_c0;
    int *d_a1, *d_b1, *d_c1;
    int size;
    int n;
};

#endif // MEGAGPU_H
