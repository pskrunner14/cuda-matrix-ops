#include <iostream>
using namespace std;

void getCudaDeviceInfo() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "GPU Device" << i << ":" << endl;
        cout << "Device name: " << prop.name << endl;
        cout << "Memory Clock Rate (KHz): " << 
            prop.memoryClockRate << endl;
        cout << "Memory Bus Width (bits): " << 
            prop.memoryBusWidth << endl;
        cout << "Peak Memory Bandwidth (GB/s): " << 
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << endl;
        cout << endl;
    }
}