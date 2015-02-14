#include <stdio.h>
#include <assert.h>
#include "FileReader.h"
#include "OSUFlowCuda.h"
#include "cp_time.h"


OSUFlowCuda::OSUFlowCuda()
{
 	initCuda(0);

	m_kernelParam.d_trace=0;
	m_kernelParam.dn_trace=0;
	m_kernelParam.d_seedPos=0;
    m_kernelParam.fullTraces = true;
    m_kernelParam.maxSteps = 100;
    m_kernelParam.stepSize = 1.f;
}

OSUFlowCuda::~OSUFlowCuda()
{
	if (m_kernelParam.d_trace) {CUDA_SAFE_CALL( cudaFree( m_kernelParam.d_trace )); m_kernelParam.d_trace=0; }
	if (m_kernelParam.dn_trace) {CUDA_SAFE_CALL( cudaFree( m_kernelParam.dn_trace )); m_kernelParam.dn_trace=0; }
	if (m_kernelParam.d_seedPos) {CUDA_SAFE_CALL( cudaFree( m_kernelParam.d_seedPos )); m_kernelParam.d_seedPos=0; }
}

void OSUFlowCuda::initCuda(int deviceID)
{
	assert(sizeof(VECTOR3)==sizeof(float3));  // ensures same type size
	CUDA_SAFE_CALL( cudaSetDevice( deviceID ));

}


void OSUFlowCuda::uploadStaticVolume(VECTOR3 *pData, const cudaExtent &volumeSize, bool bNormalize)
{
	Timer timer_copy, timer_upload;
	long size = volumeSize.width* volumeSize.height* volumeSize.depth;
	VolumeType *volume = new float4[size];

	m_volumeSize = volumeSize;

    int i;
	timer_copy.start();
	for (i=0; i<size; i++) {
		if (bNormalize) pData[i].Normalize();	//!! normalize the field
		memcpy(&volume[i], &pData[i], sizeof(VECTOR3));
        volume[i].w=0;
	}
	timer_copy.end();

	timer_upload.start();
	// create 3D array
	m_channelDesc = cudaCreateChannelDesc<VolumeType>();
	CUDA_SAFE_CALL( cudaMalloc3DArray(&md_volumeArray, &m_channelDesc, volumeSize) );

	// copy data to 3D array
	cudaMemcpy3DParms copyParams ={0};
	copyParams.dstArray = md_volumeArray;
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.kind     = cudaMemcpyHostToDevice;
	copyParams.extent	= volumeSize;
	CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams) );
	cudaDeviceSynchronize();
	timer_upload.end();

	delete[] volume;

	printf("volume size:%ld bytes, cuda preparation time:%lld, cuda upload time:%lld\n", size, timer_copy.getElapsedMS(), timer_upload.getElapsedMS());
}


void OSUFlowCuda::initSeedAry(int seeds, float3 *seedAry, KernelParam &param)
{
	param.seeds_coalesced = seeds ; // TODO
	if (param.d_trace) {CUDA_SAFE_CALL( cudaFree( param.d_trace )); param.d_trace=0; }
	if (param.dn_trace) {CUDA_SAFE_CALL( cudaFree( param.dn_trace )); param.dn_trace=0; }
	if (param.d_seedPos) {CUDA_SAFE_CALL( cudaFree( param.d_seedPos )); param.d_seedPos=0; }

    if (param.fullTraces) {
        CUDA_SAFE_CALL( cudaMalloc( &param.d_trace, sizeof(*param.d_trace) * param.seeds_coalesced * (param.maxSteps+1)) ); // +1: include the initial position
    }else{
        CUDA_SAFE_CALL( cudaMalloc( &param.d_trace, sizeof(*param.d_trace) * param.seeds_coalesced * 2/*prevent seg. fault*/) ); // only one last position
    }
	CUDA_SAFE_CALL( cudaMalloc( &param.dn_trace, sizeof(*param.dn_trace) * seeds) );
	CUDA_SAFE_CALL( cudaMalloc( &param.d_seedPos, sizeof(*param.d_seedPos) * seeds) );

	CUDA_SAFE_CALL( cudaMemcpy( param.d_seedPos, seedAry, sizeof(*param.d_seedPos) * seeds, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemset( param.dn_trace, 0, sizeof(*param.dn_trace) * seeds) );
}


void OSUFlowCuda::LoadData(const char* fname, bool bStatic, bool deferred, bool bNormalize)
{
	if (!bStatic) { printf("Non-static flowfield not supported yet!\n"); exit(1); }

	int dim[3];
	float *pData = ReadStaticDataRaw((char *)fname, dim);

	// global range
	gMin.Set(0.0,0.0,0.0);
	gMax.Set((float)(dim[0]-1), (float)(dim[1]-1), (float)(dim[2]-1));
	// local range
	lMin = gMin; lMax = gMax;

	uploadStaticVolume((VECTOR3*)pData, make_cudaExtent(dim[0], dim[1], dim[2]), bNormalize);

	delete[] pData;
	has_data=true;
}

void OSUFlowCuda::LoadData(const char* fname, bool bStatic,
		       VECTOR3 sMin, VECTOR3 sMax, bool deferred, bool bNormalize)
{
	if (!bStatic) { printf("Non-static flowfield not supported yet!\n"); exit(1); }

	int dim[3];
	float *pData = ReadStaticDataRaw((char *)fname, dim, &sMin[0], &sMax[0]);

	// global range
	gMin.Set(0.0,0.0,0.0);
	gMax.Set((float)(dim[0]-1), (float)(dim[1]-1), (float)(dim[2]-1));
	// local range
	lMin = sMin; lMax = sMax;

	uploadStaticVolume((VECTOR3*)pData, make_cudaExtent(sMax[0]-sMin[0]+1, sMax[1]-sMin[1]+1, sMax[2]-sMin[2]+1), bNormalize);

	delete[] pData;
	has_data=true;
}

bool OSUFlowCuda::GenStreamLinesRK4(list<vtListSeedTrace*>& listSeedTraces, TRACE_DIR traceDir, int maxPoints, unsigned int randomSeed)
{
	int *maxStepsAry = new int[nSeeds];
	for (int i=0; i<nSeeds; i++)
		maxStepsAry[i] = maxPoints;
	bool b = GenStreamLinesRK4(seedPtr, traceDir, nSeeds, maxStepsAry, listSeedTraces);
	delete[] maxStepsAry;
	return b;
}

bool OSUFlowCuda::GenStreamLinesRK4(VECTOR3* seeds,
                          TRACE_DIR traceDir,
                          const int seedNum,
                          const int *maxStepsAry,
                          list<vtListSeedTrace*>& listSeedTraces,
                          int64_t *seedIds,
                          list<int64_t> *listSeedIds)
{
#ifdef JCLIB
    Timer timer;
#ifdef _PROFILE
    tlog->startEvent(logGenStreamlines);
#endif
    timer.start();
#endif

    //   if (seedPtr!=NULL) delete [] seedPtr;
    nSeeds = seedNum;
    seedPtr = seeds;
    int i;
    //int threads=1;
    //if (seedNum>40) threads=4;
    //printf("seeds=%d\n", seedNum);
    #pragma omp parallel num_threads(2)
    {
        //printf("max threads=%d\n", omp_get_num_threads());
#ifdef _OPENMP
        list<vtListSeedTrace*> myListSeedTrace;
#endif
        #pragma omp for nowait schedule(dynamic, 20)
        for (i=0; i<nSeeds; i++)
        {
            //printf("thread id=%d  i=%d\n", omp_get_thread_num(), i);
            VECTOR3 pos = seeds[i];
            int j;
            vtListSeedTrace *ptrace = new vtListSeedTrace;
            ptrace->push_back(new VECTOR3(pos));
            for (j=0; j<maxStepsAry[i]; j++)
            {
                VECTOR3 outVec;
                int stat;
                float stepSize = initialStepSize;

                stat = at_phys(pos, 0, outVec);
                if (stat==-1) { ptrace->push_back(new VECTOR3(pos)); break; }
                VECTOR3 k1 = outVec * stepSize;

                stat = at_phys(pos + k1*.5f,0,  outVec);
                if (stat==-1) { ptrace->push_back(new VECTOR3(pos+k1*.5f)); break; }
                VECTOR3 k2 = outVec * stepSize;

                stat = at_phys(pos + k2*.5f,0,  outVec);
                if (stat==-1) { ptrace->push_back(new VECTOR3(pos+k2*.5f)); break; }
                VECTOR3 k3 = outVec * stepSize;

                stat = at_phys(pos + k3, 0, outVec);
                if (stat==-1) { ptrace->push_back(new VECTOR3(pos+k3)); break; }
                VECTOR3 k4 = outVec * stepSize;

                //printf("***%f %f %f %f\t", k1.x(), k2.x(), k3.x(), k4.x());
                pos = pos + (k1 + k2*2.f + k3*2.f + k4) * (1.f/6.f);

                ptrace->push_back(new VECTOR3(pos));
#ifdef _DEBUG
                printf("%f %f %f\n", pos.x(), pos.y(), pos.z());
#endif // _DEBUG
            } // j
#ifdef _OPENMP
            myListSeedTrace.push_back(ptrace);
#else
            listSeedTraces.push_back(ptrace);
#endif //_OPENMP
        } // i

#ifdef _OPENMP // combine traces
        #pragma omp critical
        listSeedTraces.splice( listSeedTraces.end(), myListSeedTrace);
        //printf("thread id %d finished\n", omp_get_thread_num());
#endif
    } // omp parallel for

    // release resource
	bool b = true;

#ifdef JCLIB
	timer.end();
	genStreamTime += timer.getElapsedMS();
#endif

#ifdef _PROFILE
    tlog->endEvent(logGenStreamlines);
#endif
	return b;
}

bool OSUFlowCuda::GenStreamLines(list<vtListSeedTrace*>& listSeedTraces, TRACE_DIR traceDir, int maxPoints, unsigned int randomSeed)
{
	return GenStreamLines(seedPtr, traceDir, nSeeds, maxPoints, listSeedTraces);

}
// seedIds, listSeedIds not implemented
bool OSUFlowCuda::GenStreamLines(VECTOR3* seeds,
        TRACE_DIR traceDir,
        const int seedNum,
		const int maxPoints,
        list<vtListSeedTrace*>& listSeedTraces,
        int64_t *seedIds,
        list<int64_t> *listSeedIds)
{
	VECTOR3 *traceAry;
	int *traceStepsAry;
	GenStreamLines_CudaNative(seeds, traceDir, seedNum, maxPoints, &traceAry, &traceStepsAry, seedIds, listSeedIds);
	int i,j;
	for (i=0; i<seedNum; i++)
	{
		std::list<VECTOR3*> *trace = new std::list<VECTOR3*>;
		for (j=0; j<traceStepsAry[i]; j++) {
			trace->push_back(new VECTOR3(traceAry[i+j*seedNum]));
    		//printf("[%f %f %f] ", traceAry[i+j*maxPoints].x(), traceAry[i+j*maxPoints].y(), traceAry[i+j*maxPoints].z());
		}
		listSeedTraces.push_back(trace);
	}

	delete[] traceAry;
	delete[] traceStepsAry;
	return true;
}

bool OSUFlowCuda::GenStreamLines_CudaNative(VECTOR3* seeds,
						TRACE_DIR traceDir,
						const int seedNum,
						const int maxPoints,
						VECTOR3 **pTraceAry,
						int **pTraceStepsAry,
						int64_t *seedIds,
						list<int64_t> *listSeedIds)
{
	printf("Run Cuda...\n");
    // init cuda
    m_kernelParam.maxSteps = maxPoints;
    m_kernelParam.stepSize = initialStepSize;
    m_kernelParam.minBound = make_float3(lMin[0], lMin[1], lMin[2]);
    m_kernelParam.maxBound = make_float3(lMax[0], lMax[1], lMax[2]);
    m_kernelParam.seeds = m_kernelParam.seeds_coalesced = seedNum;

    initSeedAry(seedNum, (float3*)seeds, m_kernelParam);

    VECTOR3 *traceAry; // output buffer
    if (m_kernelParam.fullTraces)
        traceAry = new VECTOR3[seedNum * (maxPoints+1)];
    else
        traceAry = new VECTOR3[seedNum];

    int *traceStepsAry = new int[seedNum];

#ifdef JCLIB
    Timer timer;
#ifdef _PROFILE
    tlog->startEvent(logGenStreamlines);
#endif
    timer.start();
#endif
    run_kernel(m_kernelParam, m_channelDesc, md_volumeArray);
#ifdef JCLIB
	timer.end();
#endif

#ifdef _PROFILE
    tlog->endEvent(logGenStreamlines);
#endif
	//genStreamTime += timer.getElapsedMS();

    // gather results
	if (m_kernelParam.fullTraces) {
    	CUDA_SAFE_CALL( cudaMemcpy( traceAry, m_kernelParam.d_trace, sizeof(*m_kernelParam.d_trace) * m_kernelParam.seeds_coalesced * (m_kernelParam.maxSteps+1), cudaMemcpyDeviceToHost ) );
    }else{
    	CUDA_SAFE_CALL( cudaMemcpy( traceAry, m_kernelParam.d_trace, sizeof(*m_kernelParam.d_trace) * m_kernelParam.seeds_coalesced, cudaMemcpyDeviceToHost ) );
    }
    CUDA_SAFE_CALL( cudaMemcpy( traceStepsAry, m_kernelParam.dn_trace, sizeof(*m_kernelParam.dn_trace) * m_kernelParam.seeds, cudaMemcpyDeviceToHost ));

    // return arrays
    *pTraceStepsAry = traceStepsAry;
    *pTraceAry = traceAry;

    return true;
}

int OSUFlowCuda::at_phys(const VECTOR3& pos, float t, VECTOR3 &val)
{
    m_kernelParam.minBound = make_float3(lMin[0], lMin[1], lMin[2]);
    m_kernelParam.maxBound = make_float3(lMax[0], lMax[1], lMax[2]);
    
    AtPhysResult atPhysResult =  getPhys(*(float3 *)&pos, m_kernelParam, m_channelDesc, md_volumeArray);
    val = *(VECTOR3*)&atPhysResult.vec;
    return atPhysResult.valid;
}


// be sure to load data before normalization
void OSUFlowCuda::NormalizeField(bool bLocal)
{
	if (!bLocal) printf("Warning: bLocal=false is ignored for static field\n");

	printf("Normalize field...");
	cudaExtent volumeSize = m_volumeSize;
	// bLocal not implemented
	long size = volumeSize.width* volumeSize.height* volumeSize.depth;
	VolumeType *volume = new float4[size];

	// copy data from 3D array
	cudaMemcpy3DParms copyParams ={0};
	copyParams.srcArray = md_volumeArray;
	copyParams.dstPtr   = make_cudaPitchedPtr((void *)volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.kind     = cudaMemcpyDeviceToHost;
	copyParams.extent	= volumeSize;
	CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams) );
	cudaDeviceSynchronize();

    int i;
	for (i=0; i<size; i++) {
		((VECTOR3*)&volume[i])->Normalize();	//!! normalize the field
	}
	
	// copy data to 3D array
	cudaMemcpy3DParms copyParams1 ={0};
	copyParams=copyParams1;
	copyParams.dstArray = md_volumeArray;
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.kind     = cudaMemcpyHostToDevice;
	copyParams.extent	= volumeSize;
	CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams) );
	cudaDeviceSynchronize();

	delete[] volume;
	printf("done\n");
}
