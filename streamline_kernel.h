#ifndef STREAMLINE_KERNEL_H

#include <assert.h>
#include <stdio.h>
#include <cuda.h>

///////////////
#define CUDA_SAFE_CALL(call)\
{\
cudaError err=call;\
if(cudaSuccess != err)\
{\
fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
__FILE__, __LINE__, cudaGetErrorString( err) );              \
assert(0);                                                  \
}\
}
//////////////

//typedef unsigned int  uint;
//typedef unsigned char uchar;

typedef float4 VolumeType;  // note: cuda texture does not support float3
                            // we will waste 4 byte space per vector element

// device variables
extern "C"
{
typedef struct 
{
    int seeds;
    int seeds_coalesced;
    int maxSteps;
    float stepSize;
    int fullTraces;  // false: only returns the last trace point, for graph generation
    float3 minBound;
    float3 maxBound;

    float3 *d_seedPos;
    int *dn_trace; // trace len
    float3 *d_trace; // trace record
} KernelParam;

typedef struct
{
    float3 vec;
    bool valid;
} AtPhysResult;

typedef struct
{
    int seeds;
    float stepSize;
} KernelStatus;


void run_kernel(const KernelParam &param, const cudaChannelFormatDesc &channelDesc, cudaArray *d_volumeArray);
AtPhysResult getPhys(const float3 pos, const KernelParam &param, const cudaChannelFormatDesc &channelDesc, cudaArray *d_volumeArray);

}

#endif  //STREMLINE_KERNEL_H
