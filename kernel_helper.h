#include "cutil_math.h"
///////////////
// helper functions
#if 0
extern "C"
inline __host__ __device__ float3 operator*(const float3 &a, float b)
{
return make_float3(a.x * b, a.y * b, a.z * b);
}
extern "C"
inline __host__ __device__ float3 operator/(const float3 &a, float b)
{
return make_float3(a.x / b, a.y / b, a.z / b);
}
extern "C"
inline __host__ __device__ float dot(const float3 &a, const float3 &b)
{
return a.x * b.x + a.y * b.y + a.z * b.z;
}
extern "C"
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
extern "C"
inline __host__ __device__ float3 operator+(const float3 &a, const float &b)
{
return make_float3(a.x + b, a.y + b, a.z + b);
}
extern "C"
inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
extern "C"
inline __host__ __device__ float3 normalize(const float3 &v)
{
float invLen = rsqrtf(dot(v, v));
return v * invLen;
}
extern "C"
inline __host__ __device__ void operator+=(float3 &a, const float3 &b)
{
a.x += b.x; a.y += b.y; a.z += b.z;
}

#endif
inline __host__ __device__ float3 ceil(const float3 &v)
{
	return make_float3(ceil(v.x), ceil(v.y), ceil(v.z));
}
inline __host__ __device__ double3 floorf(const double3 &v)
{
	return make_double3(floor(v.x), floor(v.y), floor(v.z));
}
inline __host__ __device__ double4 operator*(const double4 &v, const double b) 	 { return make_double4(v.x*b, v.y*b, v.z*b, v.w*b); }
inline __host__ __device__ double3 operator*(const double3 &v, const double b) 	 { return make_double3(v.x*b, v.y*b, v.z*b); }
inline __host__ __device__ double4 operator+(const double4 &v, const double b) 	 { return make_double4(v.x+b, v.y+b, v.z+b, v.w+b); }
inline __host__ __device__ double3 operator+(const double3 &v, const double b)	 { return make_double3(v.x+b, v.y+b, v.z+b); }
inline __host__ __device__ double3 operator-(const double3 &v, const double b)	 { return make_double3(v.x-b, v.y-b, v.z-b); }
inline __host__ __device__ double4 operator+(const double4 &v, const double4 &b) { return make_double4(v.x+b.x, v.y+b.y, v.z+b.z, v.w+b.w); }
inline __host__ __device__ double3 operator-(const double3 &v, const double3 &b) { return make_double3(v.x-b.x, v.y-b.y, v.z-b.z); }
inline __host__ __device__ double3 operator+(const double3 &v, const double3 &b) { return make_double3(v.x+b.x, v.y+b.y, v.z+b.z); }
inline __host__ __device__ double3 operator/(const double3 &v, const double b) { return make_double3(v.x/b, v.y/b, v.z/b); }
inline __host__ __device__ void operator+=(double3 &v, const double3 &b) { v.x+=b.x; v.y+=b.y; v.z+=b.z; }

//inline __host__ __device__ double3 operator-(const double3 &v, const float3 &b)	 { return make_double3(v.x-b.x, v.y-b.y, v.z-b.z); }


inline __device__ void assign(double4 &x, const float4 &v)	{ x = make_double4(v.x, v.y, v.z, v.w);}
inline __device__ void assign(double4 &x, const double4 &v)	{ x = v; }
inline __device__ void assign(double3 &x, const double3 &v)	{ x = v;}
inline __device__ void assign(double3 &x, const float3 &v)	{ x = make_double3(v.x, v.y, v.z);}
inline __device__ void assign(float3 &x, const double3 &v)	{ x = make_float3(v.x, v.y, v.z);}
inline __device__ void assign(float4 &x, const float4 &v)	{ x = v; }
inline __device__ void assign(float3 &x, const float3 &v)	{ x = v; }
