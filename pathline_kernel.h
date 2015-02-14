#ifndef PATHLINE_KERNEL_H

#include "streamline_kernel.h"
#include <assert.h>
#include <stdio.h>
#include <cuda.h>

//////////////


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
