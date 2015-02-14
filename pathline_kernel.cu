#include <cuda_runtime_api.h>  // C-style function interface
#include "kernel_helper.h"
#include "streamline_kernel.h"

#if 1
#define REAL	double
#define REAL4	double4
#define REAL3	double3 
#else 
#define REAL	float
#define REAL4	float4
#define REAL3	float3
#endif

texture<VolumeType, 3, cudaReadModeElementType> tex; // 3D volume
__device__ AtPhysResult d_atPhys;

//////////////
// texture lookup to get vector data
inline __device__ 
REAL4 d_texLerp3(const REAL3 &pos)
{
	REAL3 posf = floorf(pos);
	REAL3 posc = posf + 1.;
	REAL3 d,d1;
	d  = pos-posf;
	d1 = posc-pos;

	// If your indices are not normalized to [0,1], then you need to add 0.5f to each coordinate
	posc=posc+.5f;
	posf=posf+.5f;
	REAL4 data[8];
	assign( data[0], tex3D(tex, posf.x, posf.y, posf.z) );
	assign( data[1], tex3D(tex, posf.x, posf.y, posc.z) );
	assign( data[2], tex3D(tex, posf.x, posc.y, posf.z) );
	assign( data[3], tex3D(tex, posf.x, posc.y, posc.z) );
	assign( data[4], tex3D(tex, posc.x, posf.y, posf.z) );
	assign( data[5], tex3D(tex, posc.x, posf.y, posc.z) );
	assign( data[6], tex3D(tex, posc.x, posc.y, posf.z) );
	assign( data[7], tex3D(tex, posc.x, posc.y, posc.z) );
	return (
				(	(data[0]) * d1.z
				  +	(data[1]) * d.z) * d1.y 
				+(	(data[2]) * d1.z
				  +	(data[3]) * d.z) *d.y 
			) * d1.x +
			(
				(	(data[4]) * d1.z
				  +	(data[5]) * d.z) * d1.y
				+(	(data[6]) * d1.z
				  +	(data[7]) * d.z) * d.y
			) * d.x;

}

inline __device__ bool getData(const REAL3 &pos, REAL3 &vec3, const KernelParam &param)
{
    if (pos.x < param.minBound.x || pos.y < param.minBound.y || pos.z < param.minBound.z 
        || pos.x >= param.maxBound.x || pos.y >= param.maxBound.y || pos.z >= param.maxBound.z )
        return false;
#if 0 // device linear interpolation 
    // If your indices are not normalized to [0,1], then you need to add 0.5f to each coordinate
    float4 vec4 = tex3D(tex, pos.x-param.minBound.x+.5f, pos.y-param.minBound.y+.5f, pos.z-param.minBound.z+.5f);
#else // manual interpolation
	REAL3 minBound;  assign(minBound, param.minBound);
	REAL4 vec4 = d_texLerp3(pos-minBound);
#endif
    vec3.x = vec4.x; 
    vec3.y = vec4.y;
    vec3.z = vec4.z;

    return true;
}




//////////////

// main function
__global__ void d_streamline(const KernelParam param)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
if (idx >= param.seeds) return;

int i=1;
REAL3 pos;
assign(pos, param.d_seedPos[idx]);

assign(param.d_trace[idx], pos);

bool result=true;
for (i=1; i<=param.maxSteps; i++)
{
    bool r;
    REAL3 vec3;
	
	REAL stepSize = param.stepSize;
#if 1 // RK4
	
    r = getData(pos, vec3, param);
    result = result && r;
    REAL3 k1 = vec3 * stepSize;
	r = getData(pos + k1*(REAL)0.5, vec3, param);
    result = result && r;
    REAL3 k2 = vec3 * stepSize;
	r = getData(pos + k2*(REAL)0.5, vec3, param);
    result = result && r;
    REAL3 k3 = vec3 * stepSize;
	r = getData(pos + k3, vec3, param);
    result = result && r;
    REAL3 k4 = vec3 * stepSize;

    if (!result) 
        break;
    pos += (k1 + (k2 + k3)*(REAL)2 + k4) / (REAL)6;

#else
    r = getData(pos, vec3);

    if (!r) 
        break;
    pos += vec3 * param.stepSize;

#endif

    if (param.fullTraces )
        assign(param.d_trace[idx + i*param.seeds_coalesced], pos);
}
if (!param.fullTraces)
	assign(param.d_trace[idx], pos);
param.dn_trace[idx] = i;                      // return number of steps traced


}


//////////////
// (debug)
__global__ void d_getPhys(const float3 pos, const KernelParam param)
{
	REAL3 vec3;
	REAL3 rpos;
	assign(rpos, pos);
    bool r = getData(rpos, vec3, param);
    assign(d_atPhys.vec, vec3);
    d_atPhys.valid 	= r;
}

extern "C"
AtPhysResult getPhys(const float3 pos, const KernelParam &param, const cudaChannelFormatDesc &channelDesc, cudaArray *d_volumeArray)
{
// set texture parameters
tex.normalized = false;                      // access without normalized texture coordinates
tex.filterMode = cudaFilterModeLinear;      // linear interpolation
tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
tex.addressMode[1] = cudaAddressModeClamp;
tex.addressMode[2] = cudaAddressModeClamp;

// bind array to 3D texture
CUDA_SAFE_CALL( cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

//printf("pos=%f %f %f, param.min=%f %f %f\n", pos.x, pos.y, pos.z, param.minBound.x, param.minBound.y, param.minBound.z);

cudaThreadSynchronize();
d_getPhys<<<1,1>>>(pos, param);
cudaDeviceSynchronize();

AtPhysResult atPhysResult;
CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&atPhysResult, "d_atPhys", sizeof(atPhysResult),0, cudaMemcpyDeviceToHost));
//printf("atPhysResult: vec=%f %f %f, valid=%d\n", atPhysResult.vec.x, atPhysResult.vec.y, atPhysResult.vec.z, atPhysResult.valid);

return atPhysResult;
}


////////////////////////////////////////////////////////////////////////////////////

extern "C"
void run_kernel(const KernelParam &param, const cudaChannelFormatDesc &channelDesc, cudaArray *d_volumeArray)
{
	// set texture parameters
	tex.normalized = false;                      // access without normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	CUDA_SAFE_CALL( cudaBindTextureToArray(tex, d_volumeArray, channelDesc));


    cudaThreadSynchronize();
    
    dim3 blockSize( 256 ); 
    dim3 gridSize( (int)ceil(param.seeds/(float)blockSize.x) );

    d_streamline<<<gridSize, blockSize>>>(param);

    cudaDeviceSynchronize();

}
