//	File:		OSUFlowCuda.h
//
//	Author:		Jimmy
//
//	Date:		Dec 2011
//
//	Description:	Cuda computation inherited from OSUFlow
//
#ifndef OSUFLOW_CUDA_H
#define OSUFLOW_CUDA_H

#include <stdlib.h>
#include <string.h>
#include "OSUFlow.h"
////////////////////////////////
// Cuda
#include <cuda_runtime.h>  // C++-style convenience wrappers (cuda_runtime.h) built on top of the C-style functions.
#include <cuda_gl_interop.h>
#include "streamline_kernel.h"
////////////////////////////////



class OSUFlowCuda : public OSUFlow
{
	int dim[3];
	cudaChannelFormatDesc m_channelDesc;
    cudaArray *md_volumeArray;


public:
    bool fullTraces;	// false for generating graphs that only needs the ending seed position per block


    OSUFlowCuda();

    ~OSUFlowCuda();


protected:
	KernelParam m_kernelParam;
	VECTOR3 *m_traceAry;
	cudaExtent m_volumeSize;

	void initCuda(int deviceID);

	void uploadStaticVolume(VECTOR3 *pData, const cudaExtent &volumeSize, bool bNormalize);

	void initSeedAry(int seeds, float3 *seedAry, KernelParam &param);
public:
	// defer: ignored
	void LoadData(const char* fname, bool bStatic, bool deferred=false, bool normalize=false);
	void LoadData(const char* fname, bool bStatic,
			       VECTOR3 sMin, VECTOR3 sMax, bool deferred=false, bool normalize=false);

	void Boundary(VECTOR3& minB, VECTOR3& maxB) { minB = lMin;  maxB = lMax; }
	void NormalizeField(bool bLocal);

	int at_phys(const VECTOR3& pos, float t, VECTOR3 &val);

	bool GenStreamLinesRK4(VECTOR3* seeds,
                              TRACE_DIR traceDir,
                              const int seedNum,
                              const int *maxStepsAry,
                              list<vtListSeedTrace*>& listSeedTraces,
                              int64_t *seedIds=NULL,
                              list<int64_t> *listSeedIds=NULL);
	bool GenStreamLinesRK4(list<vtListSeedTrace*>& listSeedTraces, TRACE_DIR traceDir, int maxPoints, unsigned int randomSeed);

	// seedIds, listSeedIds not implemented
	bool GenStreamLines(VECTOR3* seeds,
            TRACE_DIR traceDir,
            const int seedNum,
			const int maxPoints,
            list<vtListSeedTrace*>& listSeedTraces,
            int64_t *seedIds=NULL,
            list<int64_t> *listSeedIds=NULL);

	bool GenStreamLines(list<vtListSeedTrace*>&, TRACE_DIR, int maxPoints, unsigned int randomSeed);

	bool GenStreamLines_CudaNative(VECTOR3* seeds,
							TRACE_DIR traceDir,
							const int seedNum,
							const int maxPoints,
							VECTOR3 **pTraceAry,
							int **pTraceStepsAry,
							int64_t *seedIds=NULL,
							list<int64_t> *listSeedIds=NULL);

};
#endif //#ifndef OSUFLOW_LITE_H
